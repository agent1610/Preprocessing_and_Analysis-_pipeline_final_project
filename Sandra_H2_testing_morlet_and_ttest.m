%% =========================================================================
%  H2 ANALYSIS PIPELINE: n-1 vs n+1 Sequence Comparison
%  EEG Analysis | EEGLAB | File Format (.eeg/.vhdr/.vmrk)
%
%  Hypothesis: n+1 sequences will show higher alpha power than n-1 sequences
%  (i.e., lower alpha in n-1 compared to n+1)
%
%  Pipeline:
%    1. Batch load all .vhdr files
%    2. Re-reference (average; Fz online ref restored if missing)
%    3. Identify sequence boundaries & S255 targets
%    4. Extract valid n-1 and n+1 sequences (exclude consecutive targets)
%    5. Epoch with BUFFER to avoid wavelet edge artifacts
%    6. Morlet Wavelet Transform -> crop buffer -> baseline correction (dB)
%    7. Average epochs within each file -> average sessions per participant
%    8. Paired t-test across 10 participant means (df = N-1 = 9)
%    9. Cluster-based permutation test (sign-flip, Type I error control)
%   10. Save & plot results
%
%  KEY DESIGN DECISIONS:
%
%  [1] PARTICIPANT-LEVEL AVERAGING (fixes pseudo-replication):
%      Stats run on PARTICIPANT MEANS (n=10), not pooled epochs.
%      Pooling all epochs treats each epoch as an independent brain,
%      inflating df to hundreds and driving p-values to near zero
%      for trivial effects. The correct df for 10 participants is 9.
%      Pipeline: epochs -> mean per file -> mean per participant -> t-test.
%
%  [2] BUFFER EPOCHS (fixes wavelet edge artifacts):
%      Morlet wavelets need data on BOTH SIDES of each timepoint.
%      At the edge of an epoch there is no surrounding data, causing
%      distortion ("edge artifacts"). Because the baseline sits on the
%      left edge (-750ms), without a buffer the baseline correction is
%      corrupted. Fix: extract a wider epoch (-1500ms to +4750ms),
%      run the wavelet, then CROP the 750ms buffer off both sides
%      before baselining. The analysis window (-750ms to +4000ms)
%      is then fully protected.
%
%  File structure:
%    - 40 files total: 10 participants x 4 sessions
%    - 51 sequences per run, 4 runs per session, 6 targets (S255) per run
%    - Participant ID extracted from filename via regex (see Section 1)
% =========================================================================

clc; clear; close all;

%% =========================================================================
%  SECTION 1: DEFINED PARAMETERS
% =========================================================================

% --- Paths ---
data_folder   = 'C:\Users\sandr\Downloads\sequence_hyp_eeglab_matlab';
output_folder = 'C:\Users\sandr\Downloads\sequence_hyp_eeglab_matlab\h2_output';

% --- Participant ID extraction from filename ---
% The script groups files by participant to compute participant-level means.
% Set regex pattern that extracts participant identifier.
%
%   'P\d+'     matches "P01", "P02" ... in "P01_session1.vhdr"
%   'sub-\d+'  matches "sub-01"       in "sub-01_ses-1.vhdr"
%   '^\d+'     matches leading digits in "01_session1.vhdr"
%
participant_id_pattern = 'sub-\d+';  % matches "sub-01", "sub-02" etc.
% e.g. "sub-01_ses-02_task-test_eeg_detrend_notch_ba...vhdr" -> extracts "sub-01"

% --- Target marker ---
target_marker = 'S255';

% --- Sequence timing (ms) ---
blank_pre_ms    = 750;
n_flashes       = 20;
soa_ms          = 200;
flash_period_ms = n_flashes * soa_ms;   % 4000ms
blank_post_ms   = 750;
response_ms     = 2000;

% --- Sequence boundary detection ---
% Inter-sequence gap (last flash of seq N to first flash of seq N+1) ~ 3500ms.
% Threshold at 2000ms safely identifies breaks without splitting real sequences.
seq_gap_thresh_ms = 2000;

% --- Analysis window (ms, relative to first flash of sequence) ---
analysis_start_ms = -750;    % includes 750ms blank (baseline)
analysis_end_ms   =  4000;   % through all 20 flashes

% --- Buffer for wavelet edge protection ---
% 750ms buffer absorbs edge artifacts for all frequencies >= 3Hz.
% For alpha (8-13Hz, 3 cycles): half-width ~ cycles/(2*freq) ~ 180ms -> 750ms is very safe.
buffer_ms = 750;

% Buffered epoch passed to pop_epoch (wavelet computed on this)
epoch_start_ms = analysis_start_ms - buffer_ms;   % -1500ms
epoch_end_ms   = analysis_end_ms   + buffer_ms;   % +4750ms

% Baseline correction window (applied AFTER buffer is cropped)
baseline_win = [-750 0];   % ms

% --- Re-referencing ---
% Online reference was Fz. Re-reference to average (standard for TF analysis).
% Script auto-detects whether Fz needs restoring.
ref_type = 'average';

% --- Morlet Wavelet ---
freq_range  = [1 40];    % Hz
n_freqs     = 40;
alpha_range = [8 13];    % Hz — primary band for H2
min_cycles  = 3;         % at lowest frequency
max_cycles  = 8;         % at highest frequency (linearly scaled)

% --- Statistics ---
n_permutations = 1000;
cluster_alpha  = 0.05;

% --- Output ---
if ~exist(output_folder, 'dir'), mkdir(output_folder); end


%% =========================================================================
%  SECTION 2: INITIALIZE EEGLAB & DERIVED PARAMETERS
% =========================================================================

eeglab nogui;

freqs      = logspace(log10(freq_range(1)), log10(freq_range(2)), n_freqs);
cycles_vec = linspace(min_cycles, max_cycles, n_freqs);

% Find files
files   = dir(fullfile(data_folder, '*.vhdr'));
n_files = length(files);
fprintf('\nFound %d .vhdr files\n', n_files);
if n_files == 0, error('No .vhdr files found. Check data_folder.'); end

% Extract participant IDs and build file->participant map
ppt_ids = cell(1, n_files);
for f = 1:n_files
    tok = regexp(files(f).name, participant_id_pattern, 'match');
    if isempty(tok)
        error('Cannot extract participant ID from: %s\nCheck participant_id_pattern.', files(f).name);
    end
    ppt_ids{f} = tok{1};
end
unique_ppts = unique(ppt_ids);
n_ppts      = length(unique_ppts);
fprintf('Participants: %d  |  IDs: %s\n', n_ppts, strjoin(unique_ppts, ', '));


%% =========================================================================
%  SECTION 3: MAIN LOOP — one mean power map per file per condition
% =========================================================================

file_n1_mean     = cell(1, n_files);   % {f} -> [freqs x analysis_times x chans]
file_nplus1_mean = cell(1, n_files);
valid_file       = false(1, n_files);
analysis_times_ms = [];   % filled on first successful file

for f = 1:n_files

    fprintf('\n===== FILE %d/%d: %s =====\n', f, n_files, files(f).name);

    try
        %% 3.1 LOAD
        EEG = pop_loadbv(files(f).folder, files(f).name);
        EEG = eeg_checkset(EEG);
        fprintf('  %dch | %.1fs | %dHz\n', EEG.nbchan, EEG.xmax, EEG.srate);

        %% 3.2 RE-REFERENCE TO AVERAGE
        % Fz was online reference -> flat zero channel.
        % Restore it so all 64 channels contribute equally to average ref.
        chan_labels = {EEG.chanlocs.labels};
        fz_present  = any(strcmpi(chan_labels, 'Fz'));
        if ~fz_present && ~isempty(EEG.chaninfo.nodatchans)
            EEG = pop_reref(EEG, [], 'refloc', EEG.chaninfo.nodatchans(1));
            fprintf('  Fz restored + average reference applied\n');
        else
            EEG = pop_reref(EEG, []);
            fprintf('  Average reference applied (Fz present: %d)\n', fz_present);
        end

        %% 3.3 IDENTIFY SEQUENCE BOUNDARIES
        events        = EEG.event;
        event_types   = {events.type};
        event_lat_ms  = ([events.latency] / EEG.srate) * 1000;

        is_stim     = ~strcmp(event_types,'boundary') & ...
                      ~strcmp(event_types,'New Segment') & ...
                      ~cellfun(@isempty, event_types);
        stim_idx    = find(is_stim);
        stim_lat_ms = event_lat_ms(stim_idx);

        gaps      = diff(stim_lat_ms);
        break_pts = [1, find(gaps > seq_gap_thresh_ms) + 1];
        n_seqs    = length(break_pts);
        fprintf('  Sequences: %d (expected ~204)\n', n_seqs);

        stim_seq_id = zeros(1, length(stim_idx));
        for s = 1:n_seqs
            if s < n_seqs, stim_seq_id(break_pts(s):break_pts(s+1)-1) = s;
            else,           stim_seq_id(break_pts(s):end) = s;
            end
        end

        seq_start_ms = zeros(1, n_seqs);
        for s = 1:n_seqs
            seq_start_ms(s) = event_lat_ms(stim_idx(find(stim_seq_id==s, 1)));
        end

        %% 3.4 FIND TARGET SEQUENCES
        is_target     = strcmp(event_types, target_marker);
        target_lat_ms = event_lat_ms(is_target);
        if isempty(target_lat_ms)
            warning('  No "%s" markers found. Skipping.', target_marker); continue;
        end
        fprintf('  Target triggers: %d (expected 24)\n', length(target_lat_ms));

        seq_has_target = false(1, n_seqs);
        for s = 1:n_seqs
            seq_end_ms = seq_start_ms(s) + flash_period_ms + 100;
            if any(target_lat_ms >= seq_start_ms(s)-50 & target_lat_ms <= seq_end_ms)
                seq_has_target(s) = true;
            end
        end
        target_seqs = find(seq_has_target);
        fprintf('  Target sequences: [%s]\n', num2str(target_seqs));

        %% 3.5 VALID n-1 AND n+1 (exclude around consecutive targets)
        n1_seqs = []; nplus1_seqs = [];
        for i = 1:length(target_seqs)
            tgt         = target_seqs(i);
            prev_consec = (i > 1)                   && (target_seqs(i-1) == tgt-1);
            next_consec = (i < length(target_seqs)) && (target_seqs(i+1) == tgt+1);
            if tgt > 1      && ~prev_consec, n1_seqs(end+1)     = tgt-1; end %#ok<AGROW>
            if tgt < n_seqs && ~next_consec, nplus1_seqs(end+1) = tgt+1; end %#ok<AGROW>
        end
        fprintf('  Valid n-1: %d | n+1: %d\n', length(n1_seqs), length(nplus1_seqs));
        if isempty(n1_seqs) || isempty(nplus1_seqs)
            warning('  Empty n-1 or n+1. Skipping.'); continue;
        end

        %% 3.6 EPOCH WITH BUFFER (-1500ms to +4750ms)
        % Extra 750ms on each side absorbs wavelet edge distortion.
        EEG_n1     = epoch_sequences(EEG, n1_seqs,     seq_start_ms, epoch_start_ms, epoch_end_ms);
        EEG_nplus1 = epoch_sequences(EEG, nplus1_seqs, seq_start_ms, epoch_start_ms, epoch_end_ms);

        %% 3.7 MORLET WAVELET on buffered epoch
        fprintf('  Wavelet transform...\n');
        pow_n1_buf     = compute_wavelet_power(EEG_n1,     freqs, cycles_vec);
        pow_nplus1_buf = compute_wavelet_power(EEG_nplus1, freqs, cycles_vec);

        %% 3.8 CROP BUFFER
        % Remove 750ms (= 750 samples at 1000Hz) from each end.
        % What remains is exactly the analysis window: -750ms to +4000ms.
        srate        = EEG.srate;
        buf_samps    = round(buffer_ms / 1000 * srate);
        n_buf        = size(pow_n1_buf, 2);
        crop_idx     = (buf_samps+1) : (n_buf - buf_samps);

        pow_n1     = pow_n1_buf(:,     crop_idx, :, :);
        pow_nplus1 = pow_nplus1_buf(:, crop_idx, :, :);

        % Build analysis time vector (once)
        if isempty(analysis_times_ms)
            analysis_times_ms = linspace(analysis_start_ms, analysis_end_ms, length(crop_idx));
        end

        %% 3.9 BASELINE CORRECTION (decibel)
        % 10*log10(power / mean_baseline_power)
        % Baseline = -750ms to 0ms (the blank screen before flashes).
        % Applied AFTER cropping, so the baseline window is free of edge artifacts.
        baseline_idx = analysis_times_ms >= baseline_win(1) & ...
                       analysis_times_ms <= baseline_win(2);
        pow_n1     = db_baseline_correct(pow_n1,     baseline_idx);
        pow_nplus1 = db_baseline_correct(pow_nplus1, baseline_idx);

        %% 3.10 AVERAGE EPOCHS WITHIN FILE
        % Each file contributes ONE mean map per condition.
        % This single map is then averaged with other sessions for this participant.
        file_n1_mean{f}     = mean(pow_n1,     4);   % [freqs x times x chans]
        file_nplus1_mean{f} = mean(pow_nplus1, 4);
        valid_file(f)       = true;
        fprintf('  Done.\n');

    catch ME
        warning('  ERROR: %s\n  Skipping file.', ME.message);
    end

end


%% =========================================================================
%  SECTION 4: PARTICIPANT-LEVEL AVERAGING
%
%  For each participant, average their valid session means.
%  Resulting arrays: [freqs x times x chans x n_participants]
%
%  WHY THIS MATTERS:
%  Each epoch from the same participant is NOT an independent observation.
%  Running a t-test across all pooled epochs assumes each epoch came from
%  a different brain, inflating df enormously (e.g., df~400 instead of 9).
%  This is pseudo-replication and will produce meaningless "significant"
%  p-values. By averaging to one value per participant first, we ensure
%  the t-test reflects genuine between-participant variability.
% =========================================================================

fprintf('\n===== PARTICIPANT-LEVEL AVERAGING =====\n');

first_valid = find(valid_file, 1);
[nF, nT, nC] = size(file_n1_mean{first_valid});

ppt_n1     = zeros(nF, nT, nC, n_ppts, 'single');
ppt_nplus1 = zeros(nF, nT, nC, n_ppts, 'single');
ppt_valid  = false(1, n_ppts);

for p = 1:n_ppts
    pid      = unique_ppts{p};
    fidx     = find(strcmp(ppt_ids, pid) & valid_file);
    if isempty(fidx)
        warning('  Participant %s: no valid files. Excluded.', pid); continue;
    end
    ppt_n1(:,:,:,p)     = mean(cat(4, file_n1_mean{fidx}),     4);
    ppt_nplus1(:,:,:,p) = mean(cat(4, file_nplus1_mean{fidx}), 4);
    ppt_valid(p)        = true;
    fprintf('  %s: %d session(s) averaged\n', pid, length(fidx));
end

ppt_n1     = ppt_n1(:,:,:, ppt_valid);
ppt_nplus1 = ppt_nplus1(:,:,:, ppt_valid);
n_valid    = sum(ppt_valid);
fprintf('\n  Participants in stats: %d  |  df = %d\n', n_valid, n_valid-1);


%% =========================================================================
%  SECTION 5: STATISTICS
% =========================================================================

fprintf('\n===== STATISTICS =====\n');

%% 5.1 POINT-WISE PAIRED t-TEST
% Paired because each participant contributes one n-1 AND one n+1 mean.
% df = n_participants - 1 = 9 (for 10 participants).
fprintf('  Paired t-test at every [freq x time x chan] (df=%d)...\n', n_valid-1);

t_map = zeros(nF, nT, nC);
p_map = zeros(nF, nT, nC);

for ch = 1:nC
    for fr = 1:nF
        for ti = 1:nT
            x1 = squeeze(ppt_n1(fr,ti,ch,:));
            x2 = squeeze(ppt_nplus1(fr,ti,ch,:));
            [~,p,~,stats]     = ttest(x1, x2);
            t_map(fr,ti,ch)   = stats.tstat;
            p_map(fr,ti,ch)   = p;
        end
    end
    if mod(ch,16)==0, fprintf('    ch %d/%d\n',ch,nC); end
end

%% 5.2 CLUSTER-BASED PERMUTATION TEST (alpha band)
% Uses sign-flipping of participant differences — the paired-data equivalent
% of label shuffling. Preserves the correct unit of observation (participant).
fprintf('  Cluster permutation test [%d-%dHz], %d permutations...\n', ...
        alpha_range(1), alpha_range(2), n_permutations);

alpha_freq_idx = freqs >= alpha_range(1) & freqs <= alpha_range(2);
alpha_freqs    = freqs(alpha_freq_idx);
n1_alpha       = ppt_n1(alpha_freq_idx,:,:,:);
nplus1_alpha   = ppt_nplus1(alpha_freq_idx,:,:,:);

[cluster_mask, cluster_pvals, n_sig] = ...
    cluster_permutation_test_paired(n1_alpha, nplus1_alpha, n_permutations, cluster_alpha);

fprintf('  Significant clusters: %d\n', n_sig);


%% =========================================================================
%  SECTION 6: SAVE
% =========================================================================

results.t_map             = t_map;
results.p_map             = p_map;
results.cluster_mask      = cluster_mask;
results.cluster_pvals     = cluster_pvals;
results.n_sig_clusters    = n_sig;
results.freqs             = freqs;
results.alpha_freqs       = alpha_freqs;
results.alpha_freq_idx    = alpha_freq_idx;
results.analysis_times_ms = analysis_times_ms;
results.ppt_n1            = ppt_n1;
results.ppt_nplus1        = ppt_nplus1;
results.n_valid_ppts      = n_valid;
results.unique_ppts       = unique_ppts(ppt_valid);

save(fullfile(output_folder,'H2_results.mat'), 'results', '-v7.3');
fprintf('Saved: H2_results.mat\n');


%% =========================================================================
%  SECTION 7: PLOTS
% =========================================================================

times = analysis_times_ms;
ga_n1     = squeeze(mean(ppt_n1,     4));
ga_nplus1 = squeeze(mean(ppt_nplus1, 4));

% Grand average alpha (channels averaged)
ga_n1_a = squeeze(mean(ga_n1(alpha_freq_idx,:,:),     3));
ga_np1_a = squeeze(mean(ga_nplus1(alpha_freq_idx,:,:), 3));

figure('Name','H2: Alpha Power','Position',[100 100 1200 420]);
clim_v = max(abs([ga_n1_a(:); ga_np1_a(:)]));

subplot(1,3,1); imagesc(times,alpha_freqs,ga_n1_a); axis xy; colorbar;
xlabel('Time (ms)'); ylabel('Freq (Hz)'); title('n-1 Alpha (dB)');
colormap(jet); clim([-clim_v clim_v]);

subplot(1,3,2); imagesc(times,alpha_freqs,ga_np1_a); axis xy; colorbar;
xlabel('Time (ms)'); ylabel('Freq (Hz)'); title('n+1 Alpha (dB)');
colormap(jet); clim([-clim_v clim_v]);

subplot(1,3,3); imagesc(times,alpha_freqs,ga_n1_a-ga_np1_a); axis xy; colorbar;
xlabel('Time (ms)'); ylabel('Freq (Hz)'); title('n-1 minus n+1 (dB)');
colormap(redblue_colormap(256));
saveas(gcf, fullfile(output_folder,'H2_alpha_power.png'));

% t-map
mean_t = squeeze(mean(t_map(alpha_freq_idx,:,:), 3));
figure('Name','H2: t-map','Position',[100 100 800 420]);
imagesc(times,alpha_freqs,mean_t); axis xy; colorbar;
t_lim = max(abs(mean_t(:))); clim([-t_lim t_lim]);
xlabel('Time (ms)'); ylabel('Freq (Hz)');
title(sprintf('Paired t-map (df=%d), positive = n-1 > n+1', n_valid-1));
colormap(redblue_colormap(256));
saveas(gcf, fullfile(output_folder,'H2_tmap.png'));

% Cluster mask
figure('Name','H2: Clusters','Position',[100 100 800 420]);
imagesc(times,alpha_freqs,squeeze(mean(cluster_mask,3))); axis xy; colorbar;
xlabel('Time (ms)'); ylabel('Freq (Hz)');
title(sprintf('Significant clusters (p<%.2f, permutation-corrected)', cluster_alpha));
colormap(hot);
saveas(gcf, fullfile(output_folder,'H2_clusters.png'));

fprintf('\n====== PIPELINE COMPLETE ======\n');


%% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function EEG_out = epoch_sequences(EEG, seq_nums, seq_start_ms, ep_start_ms, ep_end_ms)
    MARKER  = 'seq_epoch_temp';
    EEG_tmp = EEG;
    
    % Append directly to existing structure to prevent field-mismatch errors
    n_evts = length(EEG_tmp.event);
    added_count = 0;
    
    for i = 1:length(seq_nums)
        lat_smp = round(seq_start_ms(seq_nums(i)) / 1000 * EEG.srate);
        
        if lat_smp < 1 || lat_smp > EEG.pnts
            warning('Seq %d out of range. Skipping.', seq_nums(i)); 
            continue;
        end
        
        n_evts = n_evts + 1;
        EEG_tmp.event(n_evts).type     = MARKER;
        EEG_tmp.event(n_evts).latency  = lat_smp;
        EEG_tmp.event(n_evts).duration = 1;
        added_count = added_count + 1;
    end
    
    if added_count == 0
        error('No valid markers to epoch.'); 
    end
    
    EEG_tmp = eeg_checkset(EEG_tmp,'eventconsistency');
    EEG_out = pop_epoch(EEG_tmp, {MARKER}, [ep_start_ms/1000, ep_end_ms/1000]);
    EEG_out = eeg_checkset(EEG_out);
end


function pow = compute_wavelet_power(EEG, freqs, cycles_vec)
    [n_chans, n_times, n_epochs] = size(EEG.data);
    pow = zeros(length(freqs), n_times, n_chans, n_epochs, 'single');
    for ch = 1:n_chans
        for ep = 1:n_epochs
            sig = double(squeeze(EEG.data(ch,:,ep)));
            for fr = 1:length(freqs)
                pow(fr,:,ch,ep) = morlet_power_1d(sig, freqs(fr), cycles_vec(fr), EEG.srate);
            end
        end
    end
end


function pow = morlet_power_1d(signal, freq, n_cycles, srate)
    t_wav   = -2 : 1/srate : 2;
    sigma   = n_cycles / (2*pi*freq);
    wavelet = exp(2*1i*pi*freq.*t_wav) .* exp(-t_wav.^2/(2*sigma^2));
    wavelet = wavelet / sum(abs(wavelet));
    pow     = abs(conv(signal, wavelet, 'same')).^2;
end


function pow_db = db_baseline_correct(pow, baseline_idx)
    % pow: [freqs x times x chans x epochs]
    bl_mean = mean(pow(:, baseline_idx, :, :), 2);
    bl_mean(bl_mean == 0) = eps;
    pow_db = 10 * log10(pow ./ bl_mean);
end


function [cluster_mask, cluster_pvals, n_sig] = ...
        cluster_permutation_test_paired(data1, data2, n_perm, alpha)
    % Sign-flip permutation test for paired participant data.
    % data1, data2: [freqs x times x chans x n_participants]
    n_ppts   = size(data1, 4);
    diff_obs = data1 - data2;
    df       = n_ppts - 1;
    t_thresh = tinv(1 - alpha/2, df);

    t_obs = compute_paired_tmap(diff_obs);
    [obs_labels, obs_masses] = find_clusters_3d(t_obs, t_thresh);
    n_obs = length(obs_masses);

    if n_obs == 0
        fprintf('    No suprathreshold clusters.\n');
        cluster_mask = zeros(size(t_obs)); cluster_pvals = []; n_sig = 0; return;
    end
    fprintf('    Observed clusters: %d\n', n_obs);

    perm_max = zeros(1, n_perm);
    for p = 1:n_perm
        signs     = reshape(randi(2,1,n_ppts)*2-3, [1 1 1 n_ppts]);
        t_perm    = compute_paired_tmap(diff_obs .* signs);
        [~,masses] = find_clusters_3d(t_perm, t_thresh);
        if ~isempty(masses), perm_max(p) = max(abs(masses)); end
        if mod(p,1000)==0, fprintf('    Permutation %d/%d\n',p,n_perm); end
    end

    cluster_mask = zeros(size(t_obs)); cluster_pvals = ones(1,n_obs); n_sig = 0;
    for c = 1:n_obs
        pv = mean(perm_max >= abs(obs_masses(c)));
        cluster_pvals(c) = pv;
        if pv < alpha
            cluster_mask(obs_labels==c) = 1; n_sig = n_sig+1;
            fprintf('    Cluster %d: mass=%.2f, p=%.4f [SIGNIFICANT]\n',c,obs_masses(c),pv);
        else
            fprintf('    Cluster %d: mass=%.2f, p=%.4f\n',c,obs_masses(c),pv);
        end
    end
end


function t_map = compute_paired_tmap(diff_data)
    m  = mean(diff_data, 4);
    s  = std(diff_data, 0, 4);
    n  = size(diff_data, 4);
    se = s / sqrt(n); se(se==0) = eps;
    t_map = m ./ se;
end


function [labels, masses] = find_clusters_3d(t_map, thresh)
    % Custom 3D Connected Components (Replaces bwconncomp)
    % Uses 6-connectivity Breadth-First Search (Base MATLAB only)

    binary_map = abs(t_map) > thresh;
    [X, Y, Z]  = size(binary_map);
    labels     = zeros(X, Y, Z);
    masses     = [];
    
    cluster_id = 0;
    
    % 6-connected neighbor directional offsets
    dx = [1, -1, 0, 0, 0, 0];
    dy = [0, 0, 1, -1, 0, 0];
    dz = [0, 0, 0, 0, 1, -1];
    
    % Pre-allocate queues for speed (max possible size = total voxels)
    max_q = X * Y * Z;
    qx = zeros(max_q, 1);
    qy = zeros(max_q, 1);
    qz = zeros(max_q, 1);
    
    for z = 1:Z
        for y = 1:Y
            for x = 1:X
                % If we find a significant pixel that isn't labeled yet
                if binary_map(x, y, z) && labels(x, y, z) == 0
                    cluster_id = cluster_id + 1;
                    labels(x, y, z) = cluster_id;
                    
                    % Initialize the BFS Queue
                    head = 1;
                    tail = 1;
                    qx(tail) = x;
                    qy(tail) = y;
                    qz(tail) = z;
                    
                    current_mass = t_map(x, y, z);
                    
                    % Explore all connected neighbors
                    while head <= tail
                        cx = qx(head);
                        cy = qy(head);
                        cz = qz(head);
                        head = head + 1;
                        
                        % Check all 6 directions
                        for dir = 1:6
                            nx = cx + dx(dir);
                            ny = cy + dy(dir);
                            nz = cz + dz(dir);
                            
                            % Check grid boundaries
                            if nx > 0 && nx <= X && ny > 0 && ny <= Y && nz > 0 && nz <= Z
                                % If neighbor is significant and unlabeled, add to cluster
                                if binary_map(nx, ny, nz) && labels(nx, ny, nz) == 0
                                    labels(nx, ny, nz) = cluster_id;
                                    tail = tail + 1;
                                    qx(tail) = nx;
                                    qy(tail) = ny;
                                    qz(tail) = nz;
                                    
                                    current_mass = current_mass + t_map(nx, ny, nz);
                                end
                            end
                        end
                    end
                    % Store the final summed mass for this cluster
                    masses(cluster_id) = current_mass; %#ok<AGROW>
                end
            end
        end
    end
end

function cmap = redblue_colormap(n)
    r = [linspace(0.2,1,n/2), linspace(1,0.8,n/2)]';
    g = [linspace(0.2,1,n/2), linspace(1,0.2,n/2)]';
    b = [linspace(0.8,1,n/2), linspace(1,0.2,n/2)]';
    cmap = [r, g, b];
end