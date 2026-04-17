% ========================================================================
% BATCH UNFOLD DECONVOLUTION — P300 Oddball Paradigm
% ========================================================================
% Filename pattern:
%   sub-01_ses-01_task-test_eeg_detrend_notch_band_resample_interpbad.vhdr
%
% KEY PROCESSING ORDER:
%   1. Load continuous EEG
%   2. Rereference (on continuous data — BEFORE deconvolution)
%   3. Label + filter events
%   4. Unfold: designmat → timeexpand → glmfit → condense
%   5. Baseline correction (on beta coefficients — AFTER condense)
%   6. Extract ERPs, plot, save
%
% NOTE ON EPOCHING:
%   Traditional epoching is NOT used here. Unfold's time-expansion
%   replaces epoching entirely — uf_condense output IS your ERP in
%   [channels × time × conditions] format, already deconvolved.
%
% REQUIRES: EEGLAB + Unfold toolbox + BrainVision I/O plugin (pop_loadbv)
% ========================================================================

clearvars; close all; clc;

%% -----------------------------------------------------------------------
%  CONFIGURATION — only this section needs editing
%  -----------------------------------------------------------------------

DATA_FOLDER = 'C:\Users\sandr\Downloads\preprocessed_till_interp';

% --- Rereferencing ---
% Options:
%   'average'  — average reference (recommended for P300/ERPs)
%   'mastoid'  — linked mastoids (set channel names in MASTOID_CHANNELS)
%   'none'     — skip (only if already rereferenced in preprocessing)
REREF_TYPE       = 'average';
MASTOID_CHANNELS = {'TP9', 'TP10'};  % only used if REREF_TYPE = 'mastoid'

% --- Baseline correction window (ms) ---
% Applied to beta coefficients after uf_condense.
BASELINE_START_MS = -200;
BASELINE_END_MS   =    0;

% --- Other settings ---
PLOT_CHANNELS   = {'Cz', 'Pz', 'CPz', 'P3', 'P4', 'Oz'};
ARTIFACT_PREFIX = 'transient_bandp';
EPOCH_START     = -0.2;   % seconds (-200 ms)
EPOCH_END       =  0.8;   % seconds (+800 ms)
OUTPUT_DIR      = fullfile(DATA_FOLDER, 'unfold_results');
if ~exist(OUTPUT_DIR, 'dir'); mkdir(OUTPUT_DIR); end

%% -----------------------------------------------------------------------
%  DISCOVER ALL .vhdr FILES
%  -----------------------------------------------------------------------

vhdr_files = dir(fullfile(DATA_FOLDER, '*.vhdr'));
if isempty(vhdr_files)
    error('No .vhdr files found in:\n  %s', DATA_FOLDER);
end

fprintf('\n========================================\n');
fprintf('  Found %d .vhdr files\n', length(vhdr_files));
fprintf('  Rereferencing  : %s\n',  REREF_TYPE);
fprintf('  Baseline window: %d to %d ms\n', BASELINE_START_MS, BASELINE_END_MS);
fprintf('  Output: %s\n', OUTPUT_DIR);
fprintf('========================================\n\n');

group_results = struct();
file_count    = 0;

%% -----------------------------------------------------------------------
%  MAIN BATCH LOOP
%  -----------------------------------------------------------------------

for f = 1:length(vhdr_files)

    fname = vhdr_files(f).name;

    % Parse sub/ses IDs from filename
    tokens = regexp(fname, 'sub-(\d+)_ses-(\d+)', 'tokens');
    if isempty(tokens)
        fprintf('[SKIP] Cannot parse sub/ses from: %s\n', fname);
        continue;
    end

    sub_id  = sprintf('sub-%s', tokens{1}{1});
    ses_id  = sprintf('ses-%s', tokens{1}{2});
    file_id = sprintf('%s_%s', sub_id, ses_id);

    fprintf('\n========================================\n');
    fprintf('  Processing: %s\n', file_id);
    fprintf('========================================\n');

    out_mat = fullfile(OUTPUT_DIR, sprintf('%s_unfold.mat', file_id));

    % Skip if already processed
    if exist(out_mat, 'file')
        fprintf('  [SKIP] Already done — loading saved result.\n');
        tmp = load(out_mat);
        file_count = file_count + 1;
        group_results(file_count).data   = tmp.unfold_results;
        group_results(file_count).sub_id = sub_id;
        group_results(file_count).ses_id = ses_id;
        continue;
    end

    % ====================================================================
    %  STEP 1: Load EEG
    % ====================================================================
    fprintf('  Loading EEG...\n');
    try
        EEG = pop_loadbv(DATA_FOLDER, fname);
        EEG = eeg_checkset(EEG);
    catch err
        fprintf('  [ERROR] Load failed: %s\n', err.message); continue;
    end
    fprintf('  Loaded: %d ch | %.0f Hz | %d events\n', ...
            EEG.nbchan, EEG.srate, length(EEG.event));

    % ====================================================================
    %  STEP 2: Rereferencing
    %  MUST be done on continuous EEG BEFORE deconvolution.
    %  Rereferencing after would require applying it to the betas,
    %  which is non-standard. Doing it here on the raw signal is correct.
    % ====================================================================
    fprintf('  Rereferencing (%s)...\n', REREF_TYPE);

    switch lower(REREF_TYPE)

        case 'average'
            % Subtracts the mean of all channels at each timepoint.
            % Most common choice for P300 / ERP studies.
            EEG = pop_reref(EEG, []);
            fprintf('  Average reference applied.\n');

        case 'mastoid'
            mastoid_idx = find(ismember({EEG.chanlocs.labels}, MASTOID_CHANNELS));
            if length(mastoid_idx) ~= length(MASTOID_CHANNELS)
                fprintf('  [WARNING] Mastoid channels not found. Check MASTOID_CHANNELS.\n');
                fprintf('  Available: %s\n', strjoin({EEG.chanlocs.labels}, ', '));
                fprintf('  Skipping reref for this file.\n');
            else
                EEG = pop_reref(EEG, mastoid_idx);
                fprintf('  Mastoid reference applied (%s).\n', ...
                        strjoin(MASTOID_CHANNELS,' + '));
            end

        case 'none'
            fprintf('  Skipping rereferencing.\n');

        otherwise
            fprintf('  [WARNING] Unknown REREF_TYPE. Skipping.\n');
    end

    EEG = eeg_checkset(EEG);

    % ====================================================================
    %  STEP 3: Label Events
    % ====================================================================
    fprintf('  Labelling events...\n');

    if f == 1  % Print event types for first file only — sanity check
        fprintf('  --- Unique event types (first file only) ---\n');
        for u = 1:length(unique({EEG.event.type}))
            ut = unique({EEG.event.type}); ut = ut{u};
            fprintf('    %-25s (%d events)\n', ut, sum(strcmp({EEG.event.type}, ut)));
        end
        fprintf('  If artifact marker differs, update ARTIFACT_PREFIX.\n');
        fprintf('  --------------------------------------------\n');
    end

    [EEG.event.stim_type] = deal('Ignore');
    n_tgt = 0; n_nontgt = 0;

    for i = 1:length(EEG.event)
        raw = EEG.event(i).type;
        if isnumeric(raw); raw = num2str(raw); end
        ct = strtrim(raw);

        if strncmpi(ct, ARTIFACT_PREFIX, length(ARTIFACT_PREFIX))
            % stays Ignore
        elseif strcmp(ct, 'S255')
            EEG.event(i).stim_type = 'Target';
            n_tgt = n_tgt + 1;
        elseif length(ct) >= 2 && upper(ct(1)) == 'S'
            num_val = str2double(ct(2:end));
            if ~isnan(num_val) && num_val >= 1 && num_val <= 200
                EEG.event(i).stim_type = 'NonTarget';
                n_nontgt = n_nontgt + 1;
            end
        end
    end

    fprintf('  Events: %d Target | %d NonTarget\n', n_tgt, n_nontgt);
    if n_tgt < 10
        fprintf('  [SKIP] Too few targets (%d). Skipping file.\n', n_tgt); continue;
    end

    for i = 1:length(EEG.event)
        if ismember(EEG.event(i).stim_type, {'Target','NonTarget'})
            EEG.event(i).type = EEG.event(i).stim_type;
        end
    end
    keep_idx  = ismember({EEG.event.stim_type}, {'Target','NonTarget'});
    EEG.event = EEG.event(keep_idx);

    % ====================================================================
    %  STEP 4: Unfold Design Matrix
    % ====================================================================
    fprintf('  Building design matrix...\n');
    cfgDesign            = [];
    cfgDesign.eventtypes = {'Target', 'NonTarget'};
    cfgDesign.formula    = {'y ~ 1', 'y ~ 1'};  % cell means — one ERP per condition
    try
        EEG = uf_designmat(EEG, cfgDesign);
    catch err
        fprintf('  [ERROR] uf_designmat: %s\n', err.message); continue;
    end

    % ====================================================================
    %  STEP 5: Time-Expansion
    %  200ms SOA at 250Hz = 50 samples between stimuli.
    %  1000ms epoch = 5 SOA lengths = heavy overlap.
    %  This step builds the massive deconvolution matrix.
    % ====================================================================
    fprintf('  Time-expanding (-200ms to +800ms)...\n');
    cfgExpand            = [];
    cfgExpand.timelimits = [EPOCH_START, EPOCH_END];
    try
        EEG = uf_timeexpandDesignmat(EEG, cfgExpand);
    catch err
        fprintf('  [ERROR] uf_timeexpandDesignmat: %s\n', err.message); continue;
    end

    % ====================================================================
    %  STEP 6: Fit GLM
    % ====================================================================
    fprintf('  Fitting GLM (2–4 min)...\n');
    tic;
    try
        EEG = uf_glmfit(EEG);
    catch err
        fprintf('  [ERROR] uf_glmfit: %s\n', err.message); continue;
    end
    fprintf('  Done in %.1f sec.\n', toc);

    % ====================================================================
    %  STEP 7: Condense
    %  Reshapes raw betas into [channels x timepoints x conditions].
    %  This output IS your deconvolved ERP — no further epoching needed.
    % ====================================================================
    EEG = uf_condense(EEG);

    % ====================================================================
    %  STEP 8: Baseline Correction
    %
    %  WHY AFTER CONDENSE: The beta coefficients in EEG.unfold.beta_dc
    %  represent your ERP — not the raw EEG. Baseline correction must
    %  be applied here, to the betas, exactly as you would apply it to
    %  conventional averaged epochs.
    %
    %  HOW: For each condition and channel separately, subtract the mean
    %  amplitude during the pre-stimulus window (-200 to 0ms) from the
    %  entire time course. This sets the pre-stimulus baseline to zero,
    %  making post-stimulus deflections (like P300) clearly visible
    %  relative to a flat baseline.
    % ====================================================================
    fprintf('  Applying baseline correction (%d to %d ms)...\n', ...
            BASELINE_START_MS, BASELINE_END_MS);

    times_ms      = EEG.unfold.times * 1000;
    baseline_mask = times_ms >= BASELINE_START_MS & times_ms <= BASELINE_END_MS;

    if sum(baseline_mask) == 0
        error(['Baseline window (%d to %d ms) has no timepoints.\n' ...
               'Check that EPOCH_START <= BASELINE_START_MS.'], ...
               BASELINE_START_MS, BASELINE_END_MS);
    end

    % Apply per condition (3rd dim) and per channel (1st dim)
    for p = 1:size(EEG.unfold.beta_dc, 3)
        bl_mean = mean(EEG.unfold.beta_dc(:, baseline_mask, p), 2); % [ch x 1]
        EEG.unfold.beta_dc(:, :, p) = EEG.unfold.beta_dc(:, :, p) - bl_mean;
    end

    fprintf('  Baseline corrected using %d timepoints.\n', sum(baseline_mask));

    % ====================================================================
    %  STEP 9 (CORRECTED): Extract ERPs per Condition
    % ====================================================================
    % Unfold updates have changed this variable name across versions.
    % We will check all possible fields to make it bulletproof.
    if isfield(EEG.unfold, 'colnames')
        param_names = EEG.unfold.colnames;
    elseif isfield(EEG.unfold, 'eventtypes')
        param_names = EEG.unfold.eventtypes;
    elseif isfield(EEG.unfold, 'variablenames')
        param_names = EEG.unfold.variablenames;
    else
        % Fallback if Unfold hides the names entirely
        param_names = {'Target', 'NonTarget'}; 
    end

    % Ensure channel labels are standard strings
    chan_labels  = cellfun(@char, {EEG.chanlocs.labels}, 'UniformOutput', false);

    % Find which beta is the Target and which is the NonTarget
    tgt_idx  = find(strcmpi(param_names, 'Target') | strcmpi(param_names, 'Intercept: Target'));
    ntgt_idx = find(strcmpi(param_names, 'NonTarget') | strcmpi(param_names, 'Intercept: NonTarget'));

    % If string matching fails, we know Target was fed into cfgDesign as index 1
    if isempty(tgt_idx) || isempty(ntgt_idx)
        fprintf('  [WARNING] Using fallback indices for Beta extraction.\n');
        tgt_idx = 1;
        ntgt_idx = 2;
    end

    erp_target    = EEG.unfold.beta_dc(:, :, tgt_idx);
    erp_nontarget = EEG.unfold.beta_dc(:, :, ntgt_idx);
    erp_diff      = erp_target - erp_nontarget;

    % ====================================================================
    %  STEP 10: Save
    % ====================================================================
    unfold_results.beta_dc        = EEG.unfold.beta_dc;
    unfold_results.erp_target     = erp_target;
    unfold_results.erp_nontarget  = erp_nontarget;
    unfold_results.erp_diff       = erp_diff;
    unfold_results.times_ms       = times_ms;
    unfold_results.param_names    = param_names;
    unfold_results.tgt_idx        = tgt_idx;
    unfold_results.ntgt_idx       = ntgt_idx;
    unfold_results.chanlocs       = EEG.chanlocs;
    unfold_results.chan_labels    = chan_labels;
    unfold_results.n_target       = n_tgt;
    unfold_results.n_nontarget    = n_nontgt;
    unfold_results.sub_id         = sub_id;
    unfold_results.ses_id         = ses_id;
    unfold_results.reref_type     = REREF_TYPE;
    unfold_results.baseline_ms    = [BASELINE_START_MS, BASELINE_END_MS];
    unfold_results.source_file    = fname;

    save(out_mat, 'unfold_results', '-v7.3');
    fprintf('  Saved: %s_unfold.mat\n', file_id);

    file_count = file_count + 1;
    group_results(file_count).data   = unfold_results;
    group_results(file_count).sub_id = sub_id;
    group_results(file_count).ses_id = ses_id;

    % ====================================================================
    %  STEP 11: Plot — 7 Channels
    % ====================================================================
    fig1 = figure('Name',file_id,'NumberTitle','off','Color','w', ...
                  'Visible','off','Position',[50 50 1100 750]);
    plot_n = 0;
    for c = 1:length(PLOT_CHANNELS)
        ch_idx = find(strcmpi(chan_labels, PLOT_CHANNELS{c}));
        if isempty(ch_idx); continue; end
        plot_n = plot_n + 1;
        subplot(3,3,plot_n); hold on;
        % FIXED: Rectangle coords [200 500 500 200] and Height [-5 -5 5 5]
        patch([300 600 600 300],[-5 -5 5 5],[1.0 0.9 0.6], ...
              'FaceAlpha',0.2,'EdgeColor','none');
        plot(times_ms,erp_nontarget(ch_idx,:),'Color',[0.2 0.4 0.8],'LineWidth',1.5);
        plot(times_ms,erp_target(ch_idx,:),   'Color',[0.8 0.2 0.2],'LineWidth',2);
        plot(times_ms,erp_diff(ch_idx,:),     'Color',[0.1 0.6 0.3],'LineWidth',1.5,'LineStyle','--');
        xline(0,'k:','LineWidth',1); yline(0,'k:','LineWidth',0.5);
        hold off;
        title(PLOT_CHANNELS{c},'FontSize',11,'FontWeight','bold');
        xlabel('Time (ms)','FontSize',8); ylabel('Amp (µV)','FontSize',8);
        xlim([-200 800]); 
        ylim([-5 5]); % FIXED: Now strictly -5 to 5
        grid on; box off;
    end
    subplot(3,3,plot_n+1); axis off;
    legend([line(NaN,NaN,'Color',[0.8 0.2 0.2],'LineWidth',2), ...
            line(NaN,NaN,'Color',[0.2 0.4 0.8],'LineWidth',1.5), ...
            line(NaN,NaN,'Color',[0.1 0.6 0.3],'LineWidth',1.5,'LineStyle','--')], ...
           {'Target (S255)','NonTarget','Difference'}, ...
           'Location','best','FontSize',10,'Box','off');
    sgtitle(sprintf('%s | Deconvolved + Baselined ERPs | P300 window shaded',file_id),'FontSize',12);
    saveas(fig1, fullfile(OUTPUT_DIR, sprintf('%s_allchans.png',file_id)));
    close(fig1);

    % ====================================================================
    %  STEP 12: Pz Figure with P300 Peak Annotation
    % ====================================================================
    pz_idx = find(strcmpi(chan_labels,'Pz'));
    if ~isempty(pz_idx)
        fig2 = figure('Name','Pz','NumberTitle','off','Color','w', ...
                      'Visible','off','Position',[100 100 700 450]);
        hold on;
        % FIXED: Rectangle coords and Height
        patch([300 600 600 300],[-5 -5 5 5],[1.0 0.9 0.6], ...
              'FaceAlpha',0.25,'EdgeColor','none');
        plot(times_ms,erp_nontarget(pz_idx,:),'Color',[0.2 0.4 0.8], ...
             'LineWidth',1.8,'DisplayName','NonTarget');
        plot(times_ms,erp_target(pz_idx,:),   'Color',[0.8 0.2 0.2], ...
             'LineWidth',2.2,'DisplayName','Target (S255)');
        plot(times_ms,erp_diff(pz_idx,:),     'Color',[0.1 0.6 0.3], ...
             'LineWidth',2,'LineStyle','--','DisplayName','Difference');
        xline(0,'k:','LineWidth',1.2); yline(0,'k:','LineWidth',0.8);
        xline(300,'--','Color',[0.7 0.5 0.1],'LineWidth',0.8);
        xline(600,'--','Color',[0.7 0.5 0.1],'LineWidth',0.8);
        win_mask = times_ms >= 300 & times_ms <= 600;
        [pk_amp,pk_ri] = max(erp_diff(pz_idx,win_mask));
        pk_times = times_ms(win_mask); pk_lat = pk_times(pk_ri);
        plot(pk_lat,pk_amp,'kv','MarkerSize',9,'MarkerFaceColor',[0.1 0.6 0.3], ...
             'HandleVisibility','off');
        text(pk_lat+12,pk_amp,sprintf('%.1f µV @ %d ms',pk_amp,round(pk_lat)), ...
             'FontSize',9,'Color',[0.1 0.6 0.3]);
        hold off;
        title(sprintf('%s | Pz | Deconvolved + Baselined',file_id), ...
              'FontSize',12,'FontWeight','bold');
        xlabel('Time (ms)','FontSize',11); ylabel('Amplitude (µV)','FontSize',11);
        xlim([-200 800]);
        ylim([-5 5]); % FIXED: Now strictly -5 to 5
        legend('Location','northwest','FontSize',10,'Box','off');
        grid on; box off;
        saveas(fig2, fullfile(OUTPUT_DIR, sprintf('%s_Pz_P300.png',file_id)));
        close(fig2);
        fprintf('  P300 peak at Pz: %.2f µV @ %d ms\n', pk_amp, round(pk_lat));
    end

    fprintf('  [DONE] %s\n', file_id);

end % end batch loop

%% -----------------------------------------------------------------------
%  GROUP AVERAGE
%  -----------------------------------------------------------------------

fprintf('\n========================================\n');
fprintf('  Computing group average (%d files)...\n', file_count);
fprintf('========================================\n');

if file_count == 0; error('No files processed.'); end

ref         = group_results(1).data;
n_chans     = size(ref.erp_target, 1);
n_times     = size(ref.erp_target, 2);
times_ms    = ref.times_ms;
chan_labels  = ref.chan_labels;

all_sub_ids = {group_results.sub_id};
unique_subs = unique(all_sub_ids);
n_subs      = length(unique_subs);

sub_erp_target    = zeros(n_chans, n_times, n_subs);
sub_erp_nontarget = zeros(n_chans, n_times, n_subs);
sub_erp_diff      = zeros(n_chans, n_times, n_subs);

for s = 1:n_subs
    sub      = unique_subs{s};
    ses_mask = strcmp(all_sub_ids, sub);
    ses_data = group_results(ses_mask);
    n_ses    = length(ses_data);

    tgt_s  = zeros(n_chans, n_times, n_ses);
    ntgt_s = zeros(n_chans, n_times, n_ses);
    for k = 1:n_ses
        tgt_s(:,:,k)  = ses_data(k).data.erp_target;
        ntgt_s(:,:,k) = ses_data(k).data.erp_nontarget;
    end

    sub_erp_target(:,:,s)    = mean(tgt_s,  3);
    sub_erp_nontarget(:,:,s) = mean(ntgt_s, 3);
    sub_erp_diff(:,:,s)      = sub_erp_target(:,:,s) - sub_erp_nontarget(:,:,s);
    fprintf('  %s: %d session(s)\n', sub, n_ses);
end

grand_target    = mean(sub_erp_target,    3);
grand_nontarget = mean(sub_erp_nontarget, 3);
grand_diff      = mean(sub_erp_diff,      3);
sem_diff        = std(sub_erp_diff, 0, 3) / sqrt(n_subs);

save(fullfile(OUTPUT_DIR,'group_grand_average.mat'), ...
     'grand_target','grand_nontarget','grand_diff','sem_diff', ...
     'sub_erp_target','sub_erp_nontarget','sub_erp_diff', ...
     'times_ms','chan_labels','unique_subs','-v7.3');
fprintf('  Saved: group_grand_average.mat\n');

% Grand Average Plot
pz_idx = find(strcmpi(chan_labels,'Pz'));
if ~isempty(pz_idx)
    fig_grp = figure('Name','Grand Average','NumberTitle','off','Color','w', ...
                     'Position',[100 100 750 480]);
    hold on;
    
    % FIXED: Shrank the massive [-15 25] patch down to [-5 5] and fixed the shape
    patch([300 600 600 300],[-5 -5 5 5],[1.0 0.9 0.6], ...
          'FaceAlpha',0.2,'EdgeColor','none','HandleVisibility','off');
          
    t_fill = [times_ms, fliplr(times_ms)];
    fill(t_fill,[grand_diff(pz_idx,:)+sem_diff(pz_idx,:), ...
                 fliplr(grand_diff(pz_idx,:)-sem_diff(pz_idx,:))], ...
         [0.1 0.6 0.3],'FaceAlpha',0.15,'EdgeColor','none','HandleVisibility','off');
         
    plot(times_ms,grand_nontarget(pz_idx,:),'Color',[0.2 0.4 0.8], ...
         'LineWidth',2,'DisplayName','NonTarget');
    plot(times_ms,grand_target(pz_idx,:),   'Color',[0.8 0.2 0.2], ...
         'LineWidth',2.2,'DisplayName','Target (S255)');
    plot(times_ms,grand_diff(pz_idx,:),     'Color',[0.1 0.6 0.3], ...
         'LineWidth',2,'LineStyle','--','DisplayName','Difference (±SEM shaded)');
         
    % We add HandleVisibility='off' to hide the crosshairs and window lines from the legend
    xline(0,'k:','LineWidth',1.2, 'HandleVisibility','off'); 
    yline(0,'k:','LineWidth',0.8, 'HandleVisibility','off');
    xline(300,'--','Color',[0.7 0.5 0.1],'LineWidth',0.8, 'HandleVisibility','off');
    xline(600,'--','Color',[0.7 0.5 0.1],'LineWidth',0.8, 'HandleVisibility','off');

    win_mask = times_ms >= 300 & times_ms <= 600;
    [gp_amp,gp_ri] = max(grand_diff(pz_idx,win_mask));
    gp_t = times_ms(win_mask); gp_lat = gp_t(gp_ri);
    
    plot(gp_lat,gp_amp,'kv','MarkerSize',10,'MarkerFaceColor',[0.1 0.6 0.3], ...
         'HandleVisibility','off');
    text(gp_lat+12,gp_amp,sprintf('Grand mean P300\n%.2f µV @ %d ms',gp_amp,round(gp_lat)), ...
         'FontSize',9,'Color',[0.1 0.6 0.3]);
    hold off;

    % --- YOUR CUSTOM TITLE ---
    title(sprintf('Grand average ERP (after Deconvolution) at Pz (N = %d)',n_subs), ...
          'FontSize',13,'FontWeight','bold');
    subtitle('Baselined −200 to 0ms | P300 window 300–600ms | Green band: ±SEM');
    xlabel('Time (ms)','FontSize',12); ylabel('Amplitude (µV)','FontSize',12);
    xlim([-200 800]);
    ylim([-5 5]);
    
    legend('Location','best','FontSize',11,'Box','off');
    grid on; box off;
    saveas(fig_grp, fullfile(OUTPUT_DIR,'grand_average_Pz.png'));
    fprintf('  Saved: grand_average_Pz.png\n');
    
    % ====================================================================
    %  STEP 13: Calculate and Print Summary Statistics
    % ====================================================================
    % Calculate the mean voltage (area under the curve) across the 300-500ms window
    mean_tgt  = mean(grand_target(pz_idx, win_mask));
    mean_ntgt = mean(grand_nontarget(pz_idx, win_mask));
    mean_diff = mean(grand_diff(pz_idx, win_mask));

    fprintf('\n=======================================================\n');
    fprintf('  SUMMARY STATISTICS: Grand Average at Pz (N = %d)\n', n_subs);
    fprintf('=======================================================\n');
    fprintf('  Time Window Assessed      : 300 ms to 600 ms\n');
    fprintf('  Peak Difference Amplitude : %.2f µV\n', gp_amp);
    fprintf('  Peak Difference Latency   : %d ms\n', round(gp_lat));
    fprintf('  Mean Target Amplitude     : %.2f µV\n', mean_tgt);
    fprintf('  Mean NonTarget Amplitude  : %.2f µV\n', mean_ntgt);
    fprintf('  Mean Difference Amplitude : %.2f µV\n', mean_diff);
    fprintf('=======================================================\n\n');
end

% ====================================================================
    %  STEP 14: Inferential Statistics (Paired T-Test)
    % ====================================================================
    % Extract the mean voltage in the 300-500ms window for EACH participant
    sub_means_tgt  = squeeze(mean(sub_erp_target(pz_idx, win_mask, :), 2));
    sub_means_ntgt = squeeze(mean(sub_erp_nontarget(pz_idx, win_mask, :), 2));

    % Run a standard Paired T-Test across the 10 subjects
    [h, p_value, ci, stats] = ttest(sub_means_tgt, sub_means_ntgt);

    fprintf('  --- INFERENTIAL STATISTICS (Paired T-Test) ---\n');
    fprintf('  t(%d) = %.3f, p = %.5f\n', stats.df, stats.tstat, p_value);
    
    if p_value < 0.05
        fprintf('  RESULT: SIGNIFICANT! The P300 effect is real.\n');
    else
        fprintf('  RESULT: NOT SIGNIFICANT. (Failed to reject null hypothesis).\n');
    end
    fprintf('=======================================================\n');

    % ====================================================================
    %  STEP 15: Save Summary Statistics as an Image (.png)
    % ====================================================================
    % Create a silent, blank white figure
    fig_stat = figure('Name','Summary Stats','Color','w', ...
                      'Visible','off','Position',[150 150 550 400]);
    axis off; % Turn off the graph lines

    % Construct the text exactly as it appears in the console
    % (We use \mu to properly draw the micro symbol in MATLAB graphics)
    stat_text = sprintf([ ...
        '=======================================================\n', ...
        '  SUMMARY STATISTICS: Grand Average at Pz (N = %d)\n', ...
        '=======================================================\n', ...
        '  Time Window Assessed      : 300 ms to 600 ms\n', ...
        '  Peak Difference Amplitude : %.2f \\muV\n', ...
        '  Peak Difference Latency   : %d ms\n', ...
        '  Mean Target Amplitude     : %.2f \\muV\n', ...
        '  Mean NonTarget Amplitude  : %.2f \\muV\n', ...
        '  Mean Difference Amplitude : %.2f \\muV\n', ...
        '=======================================================\n\n', ...
        '  --- INFERENTIAL STATISTICS (Paired T-Test) ---\n', ...
        '  t(%d) = %.3f, p = %.5f\n'], ...
        n_subs, gp_amp, round(gp_lat), mean_tgt, mean_ntgt, mean_diff, ...
        stats.df, stats.tstat, p_value);

    % Add the significance conclusion
    if p_value < 0.05
        stat_text = [stat_text, '  RESULT: SIGNIFICANT! The P300 effect is real.'];
    else
        stat_text = [stat_text, '  RESULT: NOT SIGNIFICANT. (Failed to reject null).'];
    end

    % Print the text onto the center of the blank figure
    % 'Courier' font gives it that clean, perfectly aligned coding look
    text(0.05, 0.5, stat_text, 'FontName', 'Courier', 'FontSize', 11, ...
         'Interpreter', 'tex', 'VerticalAlignment', 'middle');

    % Save the image to your output folder
    saveas(fig_stat, fullfile(OUTPUT_DIR, 'grand_average_Pz_stats.png'));
    close(fig_stat);
    
    fprintf('  Saved: grand_average_Pz_stats.png\n');

% ====================================================================
%  NEW SECTION: Grand Average Plot & Stats for Cz
% ====================================================================
% Find the exact row index for the Cz electrode
cz_idx = find(strcmpi(chan_labels,'Cz'));

if ~isempty(cz_idx)
    % --- 1. The Cz Grand Average Plot ---
    fig_cz = figure('Name','Grand Average Cz','NumberTitle','off','Color','w', ...
                     'Visible','off','Position',[150 100 750 480]);
    hold on;
    
    % FIXED: Shrank the massive [-15 25] patch down to [-5 5] and fixed the shape
    patch([300 600 600 300],[-5 -5 5 5],[1.0 0.9 0.6], ...
          'FaceAlpha',0.2,'EdgeColor','none','HandleVisibility','off');
          
    t_fill = [times_ms, fliplr(times_ms)];
    fill(t_fill,[grand_diff(cz_idx,:)+sem_diff(cz_idx,:), ...
                 fliplr(grand_diff(cz_idx,:)-sem_diff(cz_idx,:))], ...
         [0.1 0.6 0.3],'FaceAlpha',0.15,'EdgeColor','none','HandleVisibility','off');
         
    plot(times_ms,grand_nontarget(cz_idx,:),'Color',[0.2 0.4 0.8], ...
         'LineWidth',2,'DisplayName','NonTarget');
    plot(times_ms,grand_target(cz_idx,:),   'Color',[0.8 0.2 0.2], ...
         'LineWidth',2.2,'DisplayName','Target (S255)');
    plot(times_ms,grand_diff(cz_idx,:),     'Color',[0.1 0.6 0.3], ...
         'LineWidth',2,'LineStyle','--','DisplayName','Difference (±SEM shaded)');
         
    xline(0,'k:','LineWidth',1.2, 'HandleVisibility','off'); 
    yline(0,'k:','LineWidth',0.8, 'HandleVisibility','off');
    xline(300,'--','Color',[0.7 0.5 0.1],'LineWidth',0.8, 'HandleVisibility','off');
    xline(600,'--','Color',[0.7 0.5 0.1],'LineWidth',0.8, 'HandleVisibility','off');

    win_mask = times_ms >= 300 & times_ms <= 600;
    [cz_amp,cz_ri] = max(grand_diff(cz_idx,win_mask));
    cz_t = times_ms(win_mask); cz_lat = cz_t(cz_ri);
    
    plot(cz_lat,cz_amp,'kv','MarkerSize',10,'MarkerFaceColor',[0.1 0.6 0.3], ...
         'HandleVisibility','off');
    text(cz_lat+12,cz_amp,sprintf('Grand mean P300\n%.2f µV @ %d ms',cz_amp,round(cz_lat)), ...
         'FontSize',9,'Color',[0.1 0.6 0.3]);
    hold off;

    title(sprintf('Grand average ERP (after Deconvolution) at Cz (N = %d)',n_subs), ...
          'FontSize',13,'FontWeight','bold');
    subtitle('Baselined −200 to 0ms | P300 window 300–600ms | Green band: ±SEM');
    xlabel('Time (ms)','FontSize',12); ylabel('Amplitude (µV)','FontSize',12);
    xlim([-200 800]);
    ylim([-5 5]);
    
    legend('Location','best','FontSize',11,'Box','off');
    grid on; box off;
    saveas(fig_cz, fullfile(OUTPUT_DIR,'grand_average_Cz.png'));
    close(fig_cz);
    
    % --- 2. Calculate Summary & Inferential Stats for Cz ---
    mean_tgt_cz  = mean(grand_target(cz_idx, win_mask));
    mean_ntgt_cz = mean(grand_nontarget(cz_idx, win_mask));
    mean_diff_cz = mean(grand_diff(cz_idx, win_mask));

    % Extract individual subject means specifically for Cz
    sub_means_tgt_cz  = squeeze(mean(sub_erp_target(cz_idx, win_mask, :), 2));
    sub_means_ntgt_cz = squeeze(mean(sub_erp_nontarget(cz_idx, win_mask, :), 2));
    
    % Run the Paired T-Test
    [h_cz, p_value_cz, ci_cz, stats_cz] = ttest(sub_means_tgt_cz, sub_means_ntgt_cz);

    % --- 3. Save Cz Summary Stats as an Image (.png) ---
    fig_stat_cz = figure('Name','Summary Stats Cz','Color','w', ...
                      'Visible','off','Position',[150 150 550 400]);
    axis off; 

    stat_text_cz = sprintf([ ...
        '=======================================================\n', ...
        '  SUMMARY STATISTICS: Grand Average at Cz (N = %d)\n', ...
        '=======================================================\n', ...
        '  Time Window Assessed      : 300 ms to 600 ms\n', ...
        '  Peak Difference Amplitude : %.2f \\muV\n', ...
        '  Peak Difference Latency   : %d ms\n', ...
        '  Mean Target Amplitude     : %.2f \\muV\n', ...
        '  Mean NonTarget Amplitude  : %.2f \\muV\n', ...
        '  Mean Difference Amplitude : %.2f \\muV\n', ...
        '=======================================================\n\n', ...
        '  --- INFERENTIAL STATISTICS (Paired T-Test) ---\n', ...
        '  t(%d) = %.3f, p = %.5f\n'], ...
        n_subs, cz_amp, round(cz_lat), mean_tgt_cz, mean_ntgt_cz, mean_diff_cz, ...
        stats_cz.df, stats_cz.tstat, p_value_cz);

    if p_value_cz < 0.05
        stat_text_cz = [stat_text_cz, '  RESULT: SIGNIFICANT! The P300 effect is real.'];
    else
        stat_text_cz = [stat_text_cz, '  RESULT: NOT SIGNIFICANT. (Failed to reject null).'];
    end

    text(0.05, 0.5, stat_text_cz, 'FontName', 'Courier', 'FontSize', 11, ...
         'Interpreter', 'tex', 'VerticalAlignment', 'middle');

    saveas(fig_stat_cz, fullfile(OUTPUT_DIR, 'grand_average_Cz_stats.png'));
    close(fig_stat_cz);
    
    fprintf('\n  [CZ ANALYSIS COMPLETE] Saved grand_average_Cz.png and grand_average_Cz_stats.png\n');
else
    fprintf('\n  [ERROR] Could not find electrode named Cz.\n');
end