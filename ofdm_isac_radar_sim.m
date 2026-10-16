%% OFDM ISAC Radar Simulasyonu
% Menzil-Doppler isleme, 2B CA-CFAR ve sabit nokta nicemleme analizi.
% MATLAB R2020a veya uzeri (tiledlayout, exportgraphics).
clear; clc; close all;

%% 0. AYARLAR
QUICK = true;    % true: hizli on kontrol | false: daha fazla Monte Carlo ve Pfa karesi
if QUICK
    N_mc = 300;  N_noise_frames = 85;   N_pfa_frames_fxp = 50;
    Pfa_op = 1e-3; % hizli on tarama hedefi
else
    N_mc = 1000; N_noise_frames = 400;  N_pfa_frames_fxp = 150;
    Pfa_op = 1e-4; % nihai tarama hedefi; daha fazla noise karesi kullanilir
end

sim_cfg.seed = 2026; rng(sim_cfg.seed);
N = 256; M = 64;                 % alt tasiyici sayisi (menzil IFFT), OFDM sembol sayisi (Doppler FFT)
BW = 16e6; fc = 5.8e9; c = 3e8;
T_useful = N / BW;               % 16 us yararli sembol suresi
T_sym = 1.25 * T_useful;         % %25 CP ile 20 us
theo_gain_dB = 10*log10(N*M);    % 42.14 dB
sim_cfg.N = N; sim_cfg.M = M; sim_cfg.BW = BW; sim_cfg.fc = fc; sim_cfg.T_sym = T_sym;
sim_cfg.N_mc = N_mc; sim_cfg.N_noise_frames = N_noise_frames; sim_cfg.N_pfa_frames_fxp = N_pfa_frames_fxp;
sim_cfg.fixed_point_model = 'block-quantized algorithmic model; not bit-true RTL';
sim_cfg.main_matrix_bits = [8 10 12 16 20];
sim_cfg.main_matrix_deltaP_dB = [35 40 45 50];
sim_cfg.main_matrix_windows = {'rectangular','Hamming range-axis'};
sim_cfg.Pfa_operating_target = Pfa_op;

% Pencereler (birim RMS: gurultu gucunu sabit tutar)
w_rect_r = ones(N, 1);
w_hamm_r = 0.54 - 0.46*cos(2*pi*(0:N-1)'/(N-1));  w_hamm_r = w_hamm_r / sqrt(mean(w_hamm_r.^2));
w_hamm_d = 0.54 - 0.46*cos(2*pi*(0:M-1)/(M-1));   w_hamm_d = w_hamm_d / sqrt(mean(w_hamm_d.^2));
W_rect_2D   = w_rect_r * ones(1, M);   % penceresiz
W_hamm_1D_r = w_hamm_r * ones(1, M);   % yalniz menzil Hamming
W_hamm_2D   = w_hamm_r * w_hamm_d;     % menzil + Doppler Hamming

% Unitary 2B donusum
run_2d_transform = @(H_in, W_2D) sqrt(N / M) * fft(ifft(H_in .* W_2D, N, 1), M, 2);

% Hedef kanal yaniti (r, v: hucre indeksi, kesirli olabilir)
n_idx = (0:N-1)'; m_idx = (0:M-1);
tone = @(r, v, A) A * exp(-1j*2*pi*(r-1)*n_idx/N) * exp(1j*2*pi*(v-1)*m_idx/M);

% Birim genlikli QPSK ve kanal kestirimi
qpsk_gen = @() exp(1j * (pi/2) * randi([0, 3], N, M));
chan_est = @(H_true, w_noise, X_tx) ((H_true .* X_tx) + w_noise) .* conj(X_tx);

%% 1. ADIM: MODEL, KANAL KESTIRIMI VE ENTEGRASYON KAZANCI
fprintf('====================================================================\n');
fprintf('ADIM 1: OFDM KANAL KESTIRIMI VE ENTEGRASYON KAZANCI DOGRULAMASI\n');
fprintf('====================================================================\n');
dR = c / (2 * BW);
dv = (c / fc) / (2 * M * T_sym);
fprintf('Menzil cozunurlugu: %.3f m | Hiz cozunurlugu: %.3f m/s\n', dR, dv);

r_test = 60; v_test = 20; SNR_in_test = -10;
X_test = qpsk_gen();
noise_test = sqrt(10^(-SNR_in_test/10)/2) * (randn(N,M) + 1j*randn(N,M));
H_hat_single = chan_est(tone(r_test, v_test, 1), noise_test, X_test);
RD_single = abs(run_2d_transform(H_hat_single, W_rect_2D)).^2;
[peak_val, max_idx] = max(RD_single(:));
[r_est, v_est] = ind2sub([N, M], max_idx);
meas_SNR_out = 10*log10(peak_val / mean(RD_single(RD_single < peak_val/10)));
fprintf('Beklenen hucre (R_bin, D_bin): (%d,%d) | Bulunan: (%d,%d)\n', r_test, v_test, r_est, v_est);
fprintf('Teorik kazanc: %.2f dB | Olculen kazanc: %.2f dB\n\n', theo_gain_dB, meas_SNR_out - SNR_in_test);

%% 2. ADIM: 2B CA-CFAR Pfa DOGRULAMASI VE AYRI KALIBRASYON/TEST
fprintf('====================================================================\n');
fprintf('ADIM 2: 2B CA-CFAR Pfa DOGRULAMASI (AYRI KALIBRASYON/TEST)\n');
fprintf('====================================================================\n');
Gr = 2; Gd = 1; Tr = 4; Td = 2;
win_r = 2*(Tr+Gr)+1; win_d = 2*(Td+Gd)+1;
guard_r = 2*Gr+1;    guard_d = 2*Gd+1;
N_train = win_r*win_d - guard_r*guard_d;          % 13*7 - 5*3 = 76 egitim hucresi

cfar_kernel = ones(win_r, win_d);
cfar_kernel(Tr+1:Tr+guard_r, Td+1:Td+guard_d) = 0;
cfar_sum_kernel = cfar_kernel;                    % yalniz egitim hucrelerini toplar
cfar_kernel = cfar_kernel / N_train;               % ortalama egitim gucu
cfar_thr = @(P, a) conv2(P, cfar_kernel, 'same') * a;

valid_mask = false(N, M);
valid_mask(Tr+Gr+1 : N-(Tr+Gr), Td+Gd+1 : M-(Td+Gd)) = true;
num_valid = sum(valid_mask(:));

ratios_rect   = zeros(num_valid, N_noise_frames);
ratios_hamm1D = zeros(num_valid, N_noise_frames);
ratios_hamm2D = zeros(num_valid, N_noise_frames);
for f = 1:N_noise_frames
    wf = (randn(N,M) + 1j*randn(N,M)) / sqrt(2);
    Pr  = abs(run_2d_transform(wf, W_rect_2D)).^2;
    Ph1 = abs(run_2d_transform(wf, W_hamm_1D_r)).^2;
    Ph2 = abs(run_2d_transform(wf, W_hamm_2D)).^2;
    Pn_r = conv2(Pr, cfar_kernel, 'same');
    Pn_h1 = conv2(Ph1, cfar_kernel, 'same');
    Pn_h2 = conv2(Ph2, cfar_kernel, 'same');
    ratios_rect(:,f)   = Pr(valid_mask)  ./ max(Pn_r(valid_mask), realmin);
    ratios_hamm1D(:,f) = Ph1(valid_mask) ./ max(Pn_h1(valid_mask), realmin);
    ratios_hamm2D(:,f) = Ph2(valid_mask) ./ max(Pn_h2(valid_mask), realmin);
end
n_cal_noise = floor(N_noise_frames/2);
cal_cols = 1:n_cal_noise;
val_cols = (n_cal_noise+1):N_noise_frames;
if isempty(val_cols)
    error('N_noise_frames en az 2 olmali; calibration ve validation kareleri ayrilamiyor.');
end

ratio_rect_cal = ratios_rect(:,cal_cols);   ratio_rect_val = ratios_rect(:,val_cols);
ratio_h1_cal = ratios_hamm1D(:,cal_cols);   ratio_h1_val = ratios_hamm1D(:,val_cols);
ratio_h2_cal = ratios_hamm2D(:,cal_cols);   ratio_h2_val = ratios_hamm2D(:,val_cols);
num_cut_val_float = numel(ratio_rect_val);
Pfa_upper_zero_float = 3 / num_cut_val_float;
fprintf('Training kareleri: %d | Bagimsiz validation kareleri: %d | Validation CUT: %.2e\n', ...
    n_cal_noise, numel(val_cols), num_cut_val_float);
fprintf('Pfa=1e-4 icin validation setinde beklenen yanlis alarm sayisi: %.1f\n', 1e-4*num_cut_val_float);

Pfa_targets = [1e-2, 5e-3, 1e-3, 1e-4];
Pfa_meas_rect = zeros(size(Pfa_targets));
Pfa_meas_hamm = zeros(size(Pfa_targets));
Pfa_meas_hamm2D = zeros(size(Pfa_targets));
Pfa_ci_rect = zeros(numel(Pfa_targets),2);
Pfa_ci_hamm = zeros(numel(Pfa_targets),2);
Pfa_ci_hamm2D = zeros(numel(Pfa_targets),2);
for p_idx = 1:length(Pfa_targets)
    Pfa_t = Pfa_targets(p_idx);
    alpha_theo = N_train * (Pfa_t^(-1/N_train) - 1);
    fa_r = sum(ratio_rect_val(:) > alpha_theo);
    fa_h1 = sum(ratio_h1_val(:) > alpha_theo);
    fa_h2 = sum(ratio_h2_val(:) > alpha_theo);
    Pfa_meas_rect(p_idx) = fa_r / num_cut_val_float;
    Pfa_meas_hamm(p_idx) = fa_h1 / num_cut_val_float;
    Pfa_meas_hamm2D(p_idx) = fa_h2 / num_cut_val_float;
    [ci_lo, ci_hi] = binom_ci95(fa_r, num_cut_val_float);   Pfa_ci_rect(p_idx,:) = [ci_lo, ci_hi];
    [ci_lo, ci_hi] = binom_ci95(fa_h1, num_cut_val_float);  Pfa_ci_hamm(p_idx,:) = [ci_lo, ci_hi];
    [ci_lo, ci_hi] = binom_ci95(fa_h2, num_cut_val_float);  Pfa_ci_hamm2D(p_idx,:) = [ci_lo, ci_hi];
    fprintf(['Hedef Pfa %.1e | Rect %.2e [%.2e, %.2e] | Hamming-1B %.2e [%.2e, %.2e] | ' ...
             'Hamming-2B %.2e [%.2e, %.2e]\n'], ...
        Pfa_t, Pfa_meas_rect(p_idx), Pfa_ci_rect(p_idx,1), Pfa_ci_rect(p_idx,2), ...
        Pfa_meas_hamm(p_idx), Pfa_ci_hamm(p_idx,1), Pfa_ci_hamm(p_idx,2), ...
        Pfa_meas_hamm2D(p_idx), Pfa_ci_hamm2D(p_idx,1), Pfa_ci_hamm2D(p_idx,2));
end

alpha_theo_op = N_train * (Pfa_op^(-1/N_train) - 1);
alpha_cal_rect   = empirical_alpha(ratio_rect_cal, Pfa_op);
alpha_cal_hamm1D = empirical_alpha(ratio_h1_cal, Pfa_op);
alpha_cal_hamm2D = empirical_alpha(ratio_h2_cal, Pfa_op);
DeltaP_theo_mask = 10*log10(N_train / alpha_cal_rect);
fprintf('\nTeorik alpha (Pfa hedefi=%.1e): %.3f\n', Pfa_op, alpha_theo_op);
fprintf('Calibration alpha (Rect/Hamming-1B/Hamming-2B): %.3f / %.3f / %.3f\n', ...
    alpha_cal_rect, alpha_cal_hamm1D, alpha_cal_hamm2D);
fprintf('Egitim penceresi kirlenmesi icin yaklasik referans guc farki: %.2f dB\n\n', DeltaP_theo_mask);

%% 3. ADIM: ASAMA A - KONUM AYRISTIRMASI
fprintf('====================================================================\n');
fprintf('ADIM 3: ASAMA A - GUC FARKI VE KONUM AYRISTIRMASI (Esit Pfa = %.1e)\n', Pfa_op);
fprintf('====================================================================\n');
SNR_strong_in = 20; A1 = 1.0; % guclu hedef giris SNRsi 20 dB
noise_pow_dyn = (A1^2) * 10^(-SNR_strong_in/10);

r1 = 60; v1 = 20;
r2_guard   = 62;   % koruma bolgesi (dR = 2 hucre <= Gr)
r2_train   = 64;   % CFAR egitim bolgesi (dR = 4 hucre)
r2_outside = 70;   % CFAR disi (dR = 10 hucre > Gr+Tr = 6)
v2_dop_near = 25;  % ayni menzil, yakin Doppler (dV = 5 hucre)

Delta_P_vec = [0, 5, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50];
nD = numel(Delta_P_vec);
Pd_guard_hamm = zeros(1,nD);
Pd_train_rect = zeros(1,nD); Pd_train_hamm1D = zeros(1,nD);
Pd_out_rect   = zeros(1,nD); Pd_out_hamm1D   = zeros(1,nD);
Pd_dop_rect   = zeros(1,nD); Pd_dop_hamm1D   = zeros(1,nD); Pd_dop_hamm2D = zeros(1,nD);
% H0 (hayalet) oranlari: zayif hedef YOK, ayni guclu hedef + ayni gurultu
G_guard_hamm = zeros(1,nD);
G_train_rect = zeros(1,nD); G_train_hamm1D = zeros(1,nD);
G_out_rect   = zeros(1,nD); G_out_hamm1D   = zeros(1,nD);
G_dop_rect   = zeros(1,nD); G_dop_hamm1D   = zeros(1,nD); G_dop_hamm2D = zeros(1,nD);

fprintf('Her hucrede: Pd (H0 hayalet orani)\n');
fprintf('dP(dB) | Koruma(H1B)  | Egitim(Rect)  Egitim(H1B) | Dis(Rect)    Dis(H1B)    | YakinDop(Rect) (H1B)  (H2B)\n');
fprintf('--------------------------------------------------------------------------------------------------------------\n');
for d_idx = 1:nD
    dP = Delta_P_vec(d_idx);
    A2 = A1 * 10^(-dP/20);
    c_gh = 0; c_tr = 0; c_th = 0; c_or = 0; c_oh = 0; c_dr = 0; c_d1 = 0; c_d2 = 0;
    g_gh = 0; g_tr = 0; g_th = 0; g_or = 0; g_oh = 0; g_dr = 0; g_d1 = 0; g_d2 = 0;
    for mc = 1:N_mc
        d1 = rand(1,2) - 0.5;  d2 = rand(1,2) - 0.5;
        w  = sqrt(noise_pow_dyn/2) * (randn(N,M) + 1j*randn(N,M));
        H1 = tone(r1 + d1(1), v1 + d1(2), A1);

        % H0: zayif hedef yok
        H0 = H1 + w;
        P0r  = abs(run_2d_transform(H0, W_rect_2D)).^2;   th0r  = cfar_thr(P0r,  alpha_cal_rect);
        P0h1 = abs(run_2d_transform(H0, W_hamm_1D_r)).^2; th0h1 = cfar_thr(P0h1, alpha_cal_hamm1D);
        P0h2 = abs(run_2d_transform(H0, W_hamm_2D)).^2;   th0h2 = cfar_thr(P0h2, alpha_cal_hamm2D);
        g_gh = g_gh + hit_resolved(P0h1, th0h1, r2_guard,   v1, r1, v1);
        g_tr = g_tr + hit_resolved(P0r,  th0r,  r2_train,   v1, r1, v1);
        g_th = g_th + hit_resolved(P0h1, th0h1, r2_train,   v1, r1, v1);
        g_or = g_or + hit_resolved(P0r,  th0r,  r2_outside, v1, r1, v1);
        g_oh = g_oh + hit_resolved(P0h1, th0h1, r2_outside, v1, r1, v1);
        g_dr = g_dr + hit_resolved(P0r,  th0r,  r1, v2_dop_near, r1, v1);
        g_d1 = g_d1 + hit_resolved(P0h1, th0h1, r1, v2_dop_near, r1, v1);
        g_d2 = g_d2 + hit_resolved(P0h2, th0h2, r1, v2_dop_near, r1, v1);

        % 1) Koruma bolgesi (dR = 2 hucre)
        H_g = H1 + tone(r2_guard + d2(1), v1 + d2(2), A2) + w;
        Ph_g = abs(run_2d_transform(H_g, W_hamm_1D_r)).^2;
        c_gh = c_gh + hit_resolved(Ph_g, cfar_thr(Ph_g, alpha_cal_hamm1D), r2_guard, v1, r1, v1);

        % 2) CFAR egitim bolgesi (dR = 4 hucre)
        H_tr = H1 + tone(r2_train + d2(1), v1 + d2(2), A2) + w;
        Pr_t = abs(run_2d_transform(H_tr, W_rect_2D)).^2;
        Ph_t = abs(run_2d_transform(H_tr, W_hamm_1D_r)).^2;
        c_tr = c_tr + hit_resolved(Pr_t, cfar_thr(Pr_t, alpha_cal_rect),   r2_train, v1, r1, v1);
        c_th = c_th + hit_resolved(Ph_t, cfar_thr(Ph_t, alpha_cal_hamm1D), r2_train, v1, r1, v1);

        % 3) CFAR penceresi disi (dR = 10 hucre)
        H_out = H1 + tone(r2_outside + d2(1), v1 + d2(2), A2) + w;
        Pr_o = abs(run_2d_transform(H_out, W_rect_2D)).^2;
        Ph_o = abs(run_2d_transform(H_out, W_hamm_1D_r)).^2;
        c_or = c_or + hit_resolved(Pr_o, cfar_thr(Pr_o, alpha_cal_rect),   r2_outside, v1, r1, v1);
        c_oh = c_oh + hit_resolved(Ph_o, cfar_thr(Ph_o, alpha_cal_hamm1D), r2_outside, v1, r1, v1);

        % 4) Ayni menzil, yakin Doppler (dV = 5 hucre)
        H_dop = H1 + tone(r1 + d2(1), v2_dop_near + d2(2), A2) + w;
        Pr_d  = abs(run_2d_transform(H_dop, W_rect_2D)).^2;
        Ph1_d = abs(run_2d_transform(H_dop, W_hamm_1D_r)).^2;
        Ph2_d = abs(run_2d_transform(H_dop, W_hamm_2D)).^2;
        c_dr = c_dr + hit_resolved(Pr_d,  cfar_thr(Pr_d,  alpha_cal_rect),   r1, v2_dop_near, r1, v1);
        c_d1 = c_d1 + hit_resolved(Ph1_d, cfar_thr(Ph1_d, alpha_cal_hamm1D), r1, v2_dop_near, r1, v1);
        c_d2 = c_d2 + hit_resolved(Ph2_d, cfar_thr(Ph2_d, alpha_cal_hamm2D), r1, v2_dop_near, r1, v1);
    end
    Pd_guard_hamm(d_idx)  = c_gh/N_mc;
    Pd_train_rect(d_idx)  = c_tr/N_mc;  Pd_train_hamm1D(d_idx) = c_th/N_mc;
    Pd_out_rect(d_idx)    = c_or/N_mc;  Pd_out_hamm1D(d_idx)   = c_oh/N_mc;
    Pd_dop_rect(d_idx)    = c_dr/N_mc;  Pd_dop_hamm1D(d_idx)   = c_d1/N_mc;  Pd_dop_hamm2D(d_idx) = c_d2/N_mc;
    G_guard_hamm(d_idx)   = g_gh/N_mc;
    G_train_rect(d_idx)   = g_tr/N_mc;  G_train_hamm1D(d_idx)  = g_th/N_mc;
    G_out_rect(d_idx)     = g_or/N_mc;  G_out_hamm1D(d_idx)    = g_oh/N_mc;
    G_dop_rect(d_idx)     = g_dr/N_mc;  G_dop_hamm1D(d_idx)    = g_d1/N_mc;  G_dop_hamm2D(d_idx)  = g_d2/N_mc;
    fprintf([' %4d  | %.2f (%.2f) | %.2f (%.2f)  %.2f (%.2f) | %.2f (%.2f)  %.2f (%.2f) | ' ...
             '%.2f (%.2f)  %.2f (%.2f)  %.2f (%.2f)\n'], dP, ...
        Pd_guard_hamm(d_idx), G_guard_hamm(d_idx), ...
        Pd_train_rect(d_idx), G_train_rect(d_idx), Pd_train_hamm1D(d_idx), G_train_hamm1D(d_idx), ...
        Pd_out_rect(d_idx), G_out_rect(d_idx), Pd_out_hamm1D(d_idx), G_out_hamm1D(d_idx), ...
        Pd_dop_rect(d_idx), G_dop_rect(d_idx), Pd_dop_hamm1D(d_idx), G_dop_hamm1D(d_idx), ...
        Pd_dop_hamm2D(d_idx), G_dop_hamm2D(d_idx));
end
fprintf('\n');

PdA_CI = struct();
PdA_names = {'guard_hamm','train_rect','train_hamm1D','out_rect','out_hamm1D', ...
             'dop_rect','dop_hamm1D','dop_hamm2D'};
PdA_values = {Pd_guard_hamm, Pd_train_rect, Pd_train_hamm1D, Pd_out_rect, ...
              Pd_out_hamm1D, Pd_dop_rect, Pd_dop_hamm1D, Pd_dop_hamm2D};
for pdi = 1:numel(PdA_names)
    ci_mat = zeros(nD,2);
    for dpi = 1:nD
        kdet = round(PdA_values{pdi}(dpi) * N_mc);
        [tmp_lo, tmp_hi] = binom_ci95(kdet, N_mc);
        ci_mat(dpi,1) = tmp_lo; ci_mat(dpi,2) = tmp_hi;
    end
    PdA_CI.(PdA_names{pdi}) = ci_mat;
end

%% 4. ADIM: ASAMA B - BLOK SEVIYESINDE SABIT NOKTA MODELI
fprintf('====================================================================\n');
fprintf('ADIM 4: SABIT NOKTA TARAMASI (BLOK-SEVIYESI MODELI)\n');
fprintf('====================================================================\n');
Delta_P_fxp = 45; % zayif hedef giris SNRsi 20-45 = -25 dB
A2_fxp = A1 * 10^(-Delta_P_fxp/20);
sim_cfg.weak_target_snr_at_fixed_point_dB = SNR_strong_in - Delta_P_fxp;
sigma_comp = sqrt(noise_pow_dyn/2);

fs_margin = 1.1;
H_cal  = tone(r1, v1, A1);
s1_cal = ifft(H_cal .* W_hamm_1D_r, N, 1) * sqrt(N);
Z_cal  = fft(s1_cal, M, 2) / sqrt(M);
clip_fn = @(x) 2^nextpow2(max(abs([real(x(:)); imag(x(:))])) * fs_margin);
cl.H = clip_fn(H_cal);  cl.W = 2^nextpow2(max(w_hamm_r) * fs_margin);
cl.s1 = clip_fn(s1_cal); cl.Z = clip_fn(Z_cal);
g1 = round(log2(cl.s1 / cl.H));  g2 = round(log2(cl.Z / cl.H));

n_cal_fxp = floor(N_pfa_frames_fxp/2);
n_val_fxp = N_pfa_frames_fxp - n_cal_fxp;
if n_cal_fxp < 1 || n_val_fxp < 1
    error('N_pfa_frames_fxp en az 2 olmali.');
end
tot_cut_fxp = n_val_fxp * num_valid;
Pfa_upper_zero = 3 / tot_cut_fxp;
fprintf('Fixed-point noise frames: calibration=%d, validation=%d; validation CUT=%.2e\n', ...
    n_cal_fxp, n_val_fxp, tot_cut_fxp);
fprintf('Sifir alarm gozleminde nominal 95%% ust sinir 3/N_CUT = %.2e\n', Pfa_upper_zero);
fprintf('Tam olcek: clip_H=%g clip_W=%g clip_s1=%g clip_Z=%g | buyume bitleri g1=%d g2=%d\n\n', ...
    cl.H, cl.W, cl.s1, cl.Z, g1, g2);

cfgs = struct('name', {}, 'Bh', {}, 'Bs', {}, 'Bz', {});
for B = [8 10 12 14 15 16 17 18]
    cfgs(end+1) = struct('name','U','Bh',B,'Bs',B,'Bz',B); %#ok<SAGROW>
end
for B = [7 8 9 10 11 12]
    cfgs(end+1) = struct('name','G','Bh',B,'Bs',B+g1,'Bz',B+g2); %#ok<SAGROW>
end
nC = numel(cfgs);

alpha_q8_shared = round(alpha_cal_hamm1D * 256) / 256;
inv_Ntrain_q = round((1/N_train) * 2^14) / 2^14;
bg_mask = false(N, M); bg_mask(120:220, 35:55) = true;

det_ref = 0; pbg_flt_acc = 0;
for mc = 1:N_mc
    d1 = rand(1,2)-0.5; d2 = rand(1,2)-0.5;
    w = sqrt(noise_pow_dyn/2) * (randn(N,M) + 1j*randn(N,M));
    H_tot = tone(r1+d1(1), v1+d1(2), A1) + tone(r2_outside+d2(1), v1+d2(2), A2_fxp) + w;
    P_flt = abs(run_2d_transform(H_tot, W_hamm_1D_r)).^2;
    det_ref = det_ref + hit_resolved(P_flt, cfar_thr(P_flt, alpha_cal_hamm1D), r2_outside, v1, r1, v1);
    pbg_flt_acc = pbg_flt_acc + mean(P_flt(bg_mask));
end
Pd_flt_ref = det_ref / N_mc;
[Pd_flt_ref_lo, Pd_flt_ref_hi] = binom_ci95(det_ref, N_mc);
Pbg_flt_ref = pbg_flt_acc / N_mc;
fprintf('Kayan nokta referans Pd = %.3f [%.3f, %.3f] | P_bg_ref = %.3e\n\n', ...
    Pd_flt_ref, Pd_flt_ref_lo, Pd_flt_ref_hi, Pbg_flt_ref);

Pd_c = nan(1,nC); Pd_ci_lo = nan(1,nC); Pd_ci_hi = nan(1,nC);
Pd_c_cal = nan(1,nC); Pd_cal_ci_lo = nan(1,nC); Pd_cal_ci_hi = nan(1,nC);
SQNR_c = nan(1,nC);
Pfa_c = nan(1,nC); Pfa_ci_lo = nan(1,nC); Pfa_ci_hi = nan(1,nC);
Pfa_c_cal = nan(1,nC); Pfa_cal_ci_lo = nan(1,nC); Pfa_cal_ci_hi = nan(1,nC);
alpha_fxp_cal = nan(1,nC); alpha_fxp_train_pfa = nan(1,nC); alpha_fxp_possible = false(1,nC);
DZ_c = zeros(1,nC); Pbg_all_c = zeros(1,nC); Pbg_nz_c = nan(1,nC);
ratio_c = zeros(1,nC); SAT_c = zeros(1,nC); cfar_ok = false(1,nC); cfar_ok_cal = false(1,nC);
zero_fa_flag = false(1,nC); zero_fa_cal_flag = false(1,nC);

fprintf('Aile Bh Bs Bz | alpha_cal | Pd ortak-alpha [CI] | SQNR dB | Pfa ortak-alpha [CI] | Pd esit-Pfa [CI] | Pfa esit-Pfa [CI] | Olu%% | Doyg%%\n');
fprintf('-------------------------------------------------------------------------------------------------------------------------------------------------------\n');
for ci = 1:nC
    cf = cfgs(ci);
    rng(sim_cfg.seed + 1000);
    lsb_Z = cl.Z / 2^(cf.Bz-1);
    Wq = quant_real(W_hamm_1D_r, cf.Bh, cl.W);

    % 4A. Format-ozel alpha kalibrasyonu
    n_cal_cells = num_valid * n_cal_fxp;
    Pcal_vals = zeros(n_cal_cells,1);
    Pncal_vals = zeros(n_cal_cells,1);
    for f = 1:n_cal_fxp
        wf = sqrt(noise_pow_dyn/2) * (randn(N,M) + 1j*randn(N,M));
        [Pq_tmp, ~, ~, Pn_tmp, lsbP_tmp, clipTh_tmp] = ...
            fxp_chain(wf, cf, Wq, cl, cfar_sum_kernel, inv_Ntrain_q, alpha_q8_shared);
        idx = (f-1)*num_valid + (1:num_valid);
        Pcal_vals(idx) = Pq_tmp(valid_mask);
        Pncal_vals(idx) = Pn_tmp(valid_mask);
    end
    [a_cal_tmp, possible_tmp, pfa_train_tmp] = ...
        calibrate_fxp_alpha(Pcal_vals, Pncal_vals, lsbP_tmp, clipTh_tmp, Pfa_op);
    alpha_fxp_cal(ci) = a_cal_tmp;
    alpha_fxp_possible(ci) = possible_tmp;
    alpha_fxp_train_pfa(ci) = pfa_train_tmp;

    % 4B. Hedefli Monte Carlo
    det_c = 0; det_cal = 0; sqnr_signal_acc = 0; sqnr_error_acc = 0; dz_acc = 0; pbg_all_acc = 0;
    pbg_nz_sum = 0; nz_cell_count = 0; sat_acc = 0;
    for mc = 1:N_mc
        d1 = rand(1,2)-0.5; d2 = rand(1,2)-0.5;
        w = sqrt(noise_pow_dyn/2) * (randn(N,M) + 1j*randn(N,M));
        H_tot = tone(r1+d1(1), v1+d1(2), A1) + tone(r2_outside+d2(1), v1+d2(2), A2_fxp) + w;
        [P_fxp, th_fxp, sat_c, Pn_fxp, lsb_P_fxp, clip_th_fxp, Z_fxp] = ...
            fxp_chain(H_tot, cf, Wq, cl, cfar_sum_kernel, inv_Ntrain_q, alpha_q8_shared);
        Z_float = run_2d_transform(H_tot, W_hamm_1D_r);
        sqnr_signal_acc = sqnr_signal_acc + sum(abs(Z_float(:)).^2);
        sqnr_error_acc = sqnr_error_acc + sum(abs(Z_float(:)-Z_fxp(:)).^2);
        det_c = det_c + hit_resolved(P_fxp, th_fxp, r2_outside, v1, r1, v1);
        if alpha_fxp_possible(ci)
            th_cal = fxp_threshold(Pn_fxp, alpha_fxp_cal(ci), lsb_P_fxp, clip_th_fxp);
            det_cal = det_cal + hit_resolved(P_fxp, th_cal, r2_outside, v1, r1, v1);
        end
        bg_vals = P_fxp(bg_mask);
        dz_acc = dz_acc + mean(bg_vals == 0);
        pbg_all_acc = pbg_all_acc + mean(bg_vals);
        nz_vals = bg_vals(bg_vals > 0);
        if ~isempty(nz_vals)
            pbg_nz_sum = pbg_nz_sum + sum(nz_vals);
            nz_cell_count = nz_cell_count + numel(nz_vals);
        end
        sat_acc = sat_acc + sat_c;
    end
    Pd_c(ci) = det_c/N_mc;
    SQNR_c(ci) = 10*log10(sqnr_signal_acc/(sqnr_error_acc + eps));
    [tmp_lo, tmp_hi] = binom_ci95(det_c, N_mc);
    Pd_ci_lo(ci) = tmp_lo; Pd_ci_hi(ci) = tmp_hi;
    if alpha_fxp_possible(ci)
        Pd_c_cal(ci) = det_cal/N_mc;
        [tmp_lo, tmp_hi] = binom_ci95(det_cal, N_mc);
        Pd_cal_ci_lo(ci) = tmp_lo; Pd_cal_ci_hi(ci) = tmp_hi;
    end

    % 4C. Validation noise frames
    fa_raw = 0; fa_cal = 0; tot = 0;
    for f = 1:n_val_fxp
        wf = sqrt(noise_pow_dyn/2) * (randn(N,M) + 1j*randn(N,M));
        [Pq, thq, ~, Pnq, lsbP_q, clipTh_q] = ...
            fxp_chain(wf, cf, Wq, cl, cfar_sum_kernel, inv_Ntrain_q, alpha_q8_shared);
        fa_raw = fa_raw + sum(Pq(valid_mask) > thq(valid_mask));
        if alpha_fxp_possible(ci)
            thq_cal = fxp_threshold(Pnq, alpha_fxp_cal(ci), lsbP_q, clipTh_q);
            fa_cal = fa_cal + sum(Pq(valid_mask) > thq_cal(valid_mask));
        end
        tot = tot + num_valid;
    end
    Pfa_c(ci) = fa_raw/tot;
    [tmp_lo, tmp_hi] = binom_ci95(fa_raw, tot);
    Pfa_ci_lo(ci) = tmp_lo; Pfa_ci_hi(ci) = tmp_hi;
    zero_fa_flag(ci) = (fa_raw == 0);
    if alpha_fxp_possible(ci)
        Pfa_c_cal(ci) = fa_cal/tot;
        [tmp_lo, tmp_hi] = binom_ci95(fa_cal, tot);
        Pfa_cal_ci_lo(ci) = tmp_lo; Pfa_cal_ci_hi(ci) = tmp_hi;
        zero_fa_cal_flag(ci) = (fa_cal == 0);
    end
    DZ_c(ci) = dz_acc/N_mc;
    Pbg_all_c(ci) = pbg_all_acc/N_mc;
    if nz_cell_count > 0, Pbg_nz_c(ci) = pbg_nz_sum/nz_cell_count; end
    ratio_c(ci) = lsb_Z/sigma_comp;
    SAT_c(ci) = sat_acc/N_mc;
    cfar_ok(ci) = (Pfa_c(ci) >= 0.5*Pfa_op) && (Pfa_c(ci) <= 2*Pfa_op);
    cfar_ok_cal(ci) = alpha_fxp_possible(ci) && ...
        (Pfa_c_cal(ci) >= 0.5*Pfa_op) && (Pfa_c_cal(ci) <= 2*Pfa_op);

    if alpha_fxp_possible(ci), a_str = sprintf('%.4f', alpha_fxp_cal(ci)); else, a_str = 'N/A'; end
    if alpha_fxp_possible(ci)
        pdcal_str = sprintf('%.2f [%.2f,%.2f]', Pd_c_cal(ci), Pd_cal_ci_lo(ci), Pd_cal_ci_hi(ci));
        pfcal_str = sprintf('%.2e [%.2e,%.2e]', Pfa_c_cal(ci), Pfa_cal_ci_lo(ci), Pfa_cal_ci_hi(ci));
    else
        pdcal_str = 'N/A (kalibre edilemedi)'; pfcal_str = 'N/A';
    end
    fprintf('%s %2d %2d %2d | %7s | Pd %.2f [%.2f,%.2f] | SQNR %5.1f dB | Pfa %.2e [%.2e,%.2e] | %s | %s | Olu%% %5.1f | Doyg%% %4.2f\n', ...
        cf.name, cf.Bh, cf.Bs, cf.Bz, a_str, Pd_c(ci), Pd_ci_lo(ci), Pd_ci_hi(ci), SQNR_c(ci), ...
        Pfa_c(ci), Pfa_ci_lo(ci), Pfa_ci_hi(ci), pdcal_str, pfcal_str, 100*DZ_c(ci), 100*SAT_c(ci));
end
idxU = find(strcmp({cfgs.name}, 'U'));  idxG = find(strcmp({cfgs.name}, 'G'));
okU = idxU(cfar_ok(idxU)); okG = idxG(cfar_ok(idxG));
rel_bg_nz = Pbg_nz_c / max(Pbg_flt_ref, eps);
rel_bg_nz(~isfinite(rel_bg_nz) | rel_bg_nz <= 0) = NaN;
fprintf('\n');

%% 5. ADIM: ANA DENEY MATRISI (BIT x PENCERE x DELTAP)
fprintf('====================================================================\n');
fprintf('ADIM 5: ANA DENEY MATRISI (BIT x PENCERE x DELTAP)\n');
fprintf('====================================================================\n');
Bits6 = [8 10 12 16 20];
DeltaP6 = [35 40 45 50];
WinNames6 = {'Dikdortgen','Hamming-menzil'};
WinMatrices6 = {W_rect_2D, W_hamm_1D_r};
AlphaFloat6 = [alpha_cal_rect, alpha_cal_hamm1D];
nB6 = numel(Bits6); nD6 = numel(DeltaP6);
if QUICK, N_mc6 = min(N_mc,120); else, N_mc6 = min(N_mc,300); end
sim_cfg.N_mc_main_matrix = N_mc6;
PfaUpper6 = 3 / (n_val_fxp * num_valid);

Pd6_raw = nan(2,nB6,nD6); Pd6_raw_lo = nan(2,nB6,nD6); Pd6_raw_hi = nan(2,nB6,nD6);
Pd6_cal = nan(2,nB6,nD6); Pd6_cal_lo = nan(2,nB6,nD6); Pd6_cal_hi = nan(2,nB6,nD6);
Pd6_float = nan(2,nB6,nD6); Pd6_float_lo = nan(2,nB6,nD6); Pd6_float_hi = nan(2,nB6,nD6);
PweakROI0_6_float = nan(2,nB6,nD6);
SQNR6_dB = nan(2,nB6,nD6);
PweakROI0_6 = nan(2,nB6,nD6); PweakROI0_6_cal = nan(2,nB6,nD6);
Pd6_incremental = nan(2,nB6,nD6); Pd6_incremental_cal = nan(2,nB6,nD6);
Pfa6_raw = nan(2,nB6); Pfa6_raw_lo = nan(2,nB6); Pfa6_raw_hi = nan(2,nB6);
Pfa6_cal = nan(2,nB6); Pfa6_cal_lo = nan(2,nB6); Pfa6_cal_hi = nan(2,nB6);
Alpha6_shared = nan(2,nB6); Alpha6_cal = nan(2,nB6);
Alpha6_possible = false(2,nB6); Pfa6_within_factor2 = false(2,nB6);
DZ6 = nan(2,nB6); SAT6 = nan(2,nB6);

fprintf('Matriste bit profilleri: '); fprintf('%d ', Bits6); fprintf('| DeltaP (dB): '); fprintf('%d ', DeltaP6); fprintf('\n');
fprintf('Pfa hedefi = %.1e | guclu hedef SNR = %d dB | ana matris Monte Carlo N=%d\n\n', Pfa_op, SNR_strong_in, N_mc6);

for wi = 1:2
    W6 = WinMatrices6{wi};
    alpha6_shared = round(AlphaFloat6(wi) * 256) / 256;
    for bi = 1:nB6
        B = Bits6(bi);
        cf6 = struct('name', sprintf('%s-%d', WinNames6{wi}, B), 'Bh', B, 'Bs', B, 'Bz', B);
        Wq6 = quant_real(W6, B, cl.W);
        Alpha6_shared(wi,bi) = alpha6_shared;

        % 5A. Format-ozel alpha
        rng(sim_cfg.seed + 6000);
        n_cal_cells6 = num_valid * n_cal_fxp;
        Pcal6 = zeros(n_cal_cells6,1);
        Pncal6 = zeros(n_cal_cells6,1);
        for f = 1:n_cal_fxp
            wf = sqrt(noise_pow_dyn/2) * (randn(N,M) + 1j*randn(N,M));
            [Ptmp, ~, ~, Pntmp, lsbPtmp, clipTtmp] = ...
                fxp_chain(wf, cf6, Wq6, cl, cfar_sum_kernel, inv_Ntrain_q, alpha6_shared);
            idxf = (f-1)*num_valid + (1:num_valid);
            Pcal6(idxf) = Ptmp(valid_mask);
            Pncal6(idxf) = Pntmp(valid_mask);
        end
        [a6, a6_possible, pfa_train6] = ...
            calibrate_fxp_alpha(Pcal6, Pncal6, lsbPtmp, clipTtmp, Pfa_op);
        Alpha6_cal(wi,bi) = a6;
        Alpha6_possible(wi,bi) = a6_possible;

        % 5B. Bagimsiz validation gurultusu
        rng(sim_cfg.seed + 6002);
        fa_shared = 0; fa_cal = 0; ncut6 = 0;
        zero_cells6 = 0; all_cells6 = 0; sat_acc6 = 0;
        for f = 1:n_val_fxp
            wf = sqrt(noise_pow_dyn/2) * (randn(N,M) + 1j*randn(N,M));
            [Pq, thq, satq, Pnq, lsbPq, clipTq] = ...
                fxp_chain(wf, cf6, Wq6, cl, cfar_sum_kernel, inv_Ntrain_q, alpha6_shared);
            fa_shared = fa_shared + sum(Pq(valid_mask) > thq(valid_mask));
            if a6_possible
                thcal = fxp_threshold(Pnq, a6, lsbPq, clipTq);
                fa_cal = fa_cal + sum(Pq(valid_mask) > thcal(valid_mask));
            end
            zero_cells6 = zero_cells6 + sum(Pq(valid_mask) == 0);
            all_cells6 = all_cells6 + num_valid;
            sat_acc6 = sat_acc6 + satq;
            ncut6 = ncut6 + num_valid;
        end
        Pfa6_raw(wi,bi) = fa_shared/ncut6;
        [Pfa6_raw_lo(wi,bi), Pfa6_raw_hi(wi,bi)] = binom_ci95(fa_shared,ncut6);
        if a6_possible
            Pfa6_cal(wi,bi) = fa_cal/ncut6;
            [Pfa6_cal_lo(wi,bi), Pfa6_cal_hi(wi,bi)] = binom_ci95(fa_cal,ncut6);
        end
        Pfa6_within_factor2(wi,bi) = a6_possible && isfinite(Pfa6_cal(wi,bi)) && ...
            Pfa6_cal(wi,bi) >= 0.5*Pfa_op && Pfa6_cal(wi,bi) <= 2*Pfa_op;
        DZ6(wi,bi) = zero_cells6/max(all_cells6,1);
        SAT6(wi,bi) = sat_acc6/max(n_val_fxp,1);

        % 5C. Hedefli deney (eslesmis denemeler)
        for di = 1:nD6
            dP6 = DeltaP6(di);
            Aweak6 = A1 * 10^(-dP6/20);
            rng(sim_cfg.seed + 7000 + di);
            hit_shared = 0; hit_cal = 0; hit0_shared = 0; hit0_cal = 0;
            hit_incremental = 0; hit_incremental_cal = 0;
            hit_float = 0; hit0_float = 0;
            sqnr_signal_acc = 0; sqnr_error_acc = 0;
            for mc = 1:N_mc6
                d1 = rand(1,2)-0.5; d2 = rand(1,2)-0.5;
                w = sqrt(noise_pow_dyn/2) * (randn(N,M) + 1j*randn(N,M));
                H1mc = tone(r1+d1(1), v1+d1(2), A1);
                Hweak = tone(r2_outside+d2(1), v1+d2(2), Aweak6);
                Htot = H1mc + Hweak + w;
                [Pq, thq, ~, Pnq, lsbPq, clipTq, Zq] = ...
                    fxp_chain(Htot, cf6, Wq6, cl, cfar_sum_kernel, inv_Ntrain_q, alpha6_shared);
                Zfloat6 = run_2d_transform(Htot, W6);
                sqnr_signal_acc = sqnr_signal_acc + sum(abs(Zfloat6(:)).^2);
                sqnr_error_acc = sqnr_error_acc + sum(abs(Zfloat6(:)-Zq(:)).^2);
                hit_h1 = hit_resolved(Pq, thq, r2_outside, v1, r1, v1);
                hit_shared = hit_shared + hit_h1;

                % Kayan nokta referansi
                Pfl6 = abs(Zfloat6).^2;
                hit_float = hit_float + hit_resolved(Pfl6, cfar_thr(Pfl6, AlphaFloat6(wi)), r2_outside, v1, r1, v1);

                % H0: zayif hedef yok
                H0 = H1mc + w;
                [P0, th0, ~, Pn0, lsbP0, clipT0] = ...
                    fxp_chain(H0, cf6, Wq6, cl, cfar_sum_kernel, inv_Ntrain_q, alpha6_shared);
                hit_h0 = hit_resolved(P0, th0, r2_outside, v1, r1, v1);
                hit0_shared = hit0_shared + hit_h0;
                hit_incremental = hit_incremental + (hit_h1 && ~hit_h0);
                P0fl = abs(run_2d_transform(H0, W6)).^2;
                hit0_float = hit0_float + hit_resolved(P0fl, cfar_thr(P0fl, AlphaFloat6(wi)), r2_outside, v1, r1, v1);

                if a6_possible
                    th_cal = fxp_threshold(Pnq, a6, lsbPq, clipTq);
                    hit_cal_trial = hit_resolved(Pq, th_cal, r2_outside, v1, r1, v1);
                    hit_cal = hit_cal + hit_cal_trial;
                    th0_cal = fxp_threshold(Pn0, a6, lsbP0, clipT0);
                    hit_h0_cal = hit_resolved(P0, th0_cal, r2_outside, v1, r1, v1);
                    hit0_cal = hit0_cal + hit_h0_cal;
                    hit_incremental_cal = hit_incremental_cal + (hit_cal_trial && ~hit_h0_cal);
                end
            end
            Pd6_raw(wi,bi,di) = hit_shared/N_mc6;
            [Pd6_raw_lo(wi,bi,di), Pd6_raw_hi(wi,bi,di)] = binom_ci95(hit_shared,N_mc6);
            Pd6_float(wi,bi,di) = hit_float/N_mc6;
            [Pd6_float_lo(wi,bi,di), Pd6_float_hi(wi,bi,di)] = binom_ci95(hit_float,N_mc6);
            PweakROI0_6_float(wi,bi,di) = hit0_float/N_mc6;
            SQNR6_dB(wi,bi,di) = 10*log10(sqnr_signal_acc/(sqnr_error_acc + eps));
            PweakROI0_6(wi,bi,di) = hit0_shared/N_mc6;
            Pd6_incremental(wi,bi,di) = hit_incremental/N_mc6;
            if a6_possible
                Pd6_cal(wi,bi,di) = hit_cal/N_mc6;
                [Pd6_cal_lo(wi,bi,di), Pd6_cal_hi(wi,bi,di)] = binom_ci95(hit_cal,N_mc6);
                PweakROI0_6_cal(wi,bi,di) = hit0_cal/N_mc6;
                Pd6_incremental_cal(wi,bi,di) = hit_incremental_cal/N_mc6;
            end
            fprintf(['%-15s B=%2d dP=%2d | Pd kayan=%.2f | Pd sabit(esit Pfa)=%5s | ' ...
                     'DeltaPd=%6s | H0 kayan=%.2f | SQNR=%.1f dB\n'], ...
                WinNames6{wi}, B, dP6, Pd6_float(wi,bi,di), num_or_na(Pd6_cal(wi,bi,di)), ...
                num_or_na(Pd6_float(wi,bi,di) - Pd6_cal(wi,bi,di)), PweakROI0_6_float(wi,bi,di), SQNR6_dB(wi,bi,di));
        end
        if a6_possible
            fprintf('  alpha ortak=%.4f, alpha format-ozel=%.4f, Pfa ortak=%.2e, Pfa kalibre=%.2e (egitim Pfa=%.2e)\n', ...
                alpha6_shared, a6, Pfa6_raw(wi,bi), Pfa6_cal(wi,bi), pfa_train6);
        else
            fprintf('  alpha ortak=%.4f, format-ozel alpha bulunamadi (KALIBRE EDILEMEDI); Pfa ortak=%.2e\n', ...
                alpha6_shared, Pfa6_raw(wi,bi));
        end
    end
end
% Nicemleme kaybi
DeltaPd6_cal = Pd6_float - Pd6_cal;
DeltaPd6_raw = Pd6_float - Pd6_raw;

fprintf('\n');

%% GRAFIK 1: ASAMA A + Pfa DOGRULAMASI
fig1 = figure('Color', 'w', 'Position', [30, 30, 1200, 720]);
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = nexttile;
H_d = tone(r1+0.35, v1-0.25, A1) + tone(r2_outside-0.2, v1+0.15, A2_fxp) + tone(r1+0.1, v2_dop_near-0.2, A2_fxp);
w_demo = sqrt(noise_pow_dyn/2) * (randn(N,M) + 1j*randn(N,M));
RD_demo = 10*log10(abs(run_2d_transform(H_d + w_demo, W_hamm_1D_r)).^2 + eps);
imagesc(1:M, 1:100, RD_demo(1:100, :)); axis xy; cb = colorbar; cb.Color = 'k'; hold on;
hm1 = plot(v1, r1, 'ro', 'MarkerSize', 9, 'LineWidth', 2);
hm2 = plot(v1, r2_outside, 'wx', 'MarkerSize', 9, 'LineWidth', 2);
hm3 = plot(v2_dop_near, r1, 'ms', 'MarkerSize', 9, 'LineWidth', 2);
title('1) Range-Doppler Haritasi (Menzil Hamming, dB)', 'Color', 'k', 'FontSize', 10);
xlabel('Doppler Hucresi'); ylabel('Menzil Hucresi');
legend([hm1 hm2 hm3], {'Guclu hedef (goreli 0 dB)', sprintf('Zayif-menzil (goreli -%d dB)', Delta_P_fxp), ...
    sprintf('Zayif-yakin Doppler (goreli -%d dB)', Delta_P_fxp)}, ...
    'Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'FontSize', 8);
style_axes(ax);

ax = nexttile;
pfa_plot_rect = Pfa_meas_rect; pfa_plot_rect(pfa_plot_rect==0) = Pfa_upper_zero_float;
pfa_plot_h1 = Pfa_meas_hamm; pfa_plot_h1(pfa_plot_h1==0) = Pfa_upper_zero_float;
pfa_plot_h2 = Pfa_meas_hamm2D; pfa_plot_h2(pfa_plot_h2==0) = Pfa_upper_zero_float;
loglog(Pfa_targets, Pfa_targets, 'k--', 'LineWidth', 1.6); hold on;
loglog(Pfa_targets, pfa_plot_rect, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 6);
loglog(Pfa_targets, pfa_plot_h1,   'rs-', 'LineWidth', 1.5, 'MarkerSize', 6);
loglog(Pfa_targets, pfa_plot_h2,   'm^-', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on; xlabel('Teorik Hedef P_{fa}'); ylabel('Olculen P_{fa} (Monte Carlo)');
title('2) 2B CA-CFAR Pfa (teorik alpha, bagimsiz validation)', 'Color', 'k', 'FontSize', 10);
legend({'Ideal Teorik', 'Dikdortgen', 'Hamming (Menzil)', 'Hamming (Menzil+Doppler)'}, ...
    'Location', 'northwest', 'TextColor', 'k', 'Color', 'w', 'FontSize', 8);
style_axes(ax);

% Panel 3: menzil konumlari
ax = nexttile; hold on;
h3 = gobjects(1,6);
h3(1) = plot(Delta_P_vec, Pd_out_hamm1D,   'g-o',  'LineWidth', 1.8, 'MarkerFaceColor', 'g', 'MarkerSize', 5);
h3(2) = plot(Delta_P_vec, Pd_out_rect,     'b--s', 'LineWidth', 1.6, 'MarkerFaceColor', 'b', 'MarkerSize', 5);
h3(3) = plot(Delta_P_vec, Pd_train_rect,   'm:d',  'LineWidth', 1.6, 'MarkerFaceColor', 'm', 'MarkerSize', 5);
h3(4) = plot(Delta_P_vec, Pd_train_hamm1D, 'r-.^', 'LineWidth', 1.6, 'MarkerFaceColor', 'r', 'MarkerSize', 5);
h3(5) = plot(Delta_P_vec, Pd_guard_hamm,   'k-x',  'LineWidth', 1.4, 'MarkerSize', 6);
plot(Delta_P_vec, G_out_hamm1D, 'g:', 'LineWidth', 1.0);
plot(Delta_P_vec, G_out_rect,   'b:', 'LineWidth', 1.0);
plot(Delta_P_vec, G_guard_hamm, 'k:', 'LineWidth', 1.0);
h3(6) = plot(nan, nan, ':', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.0);
xline(DeltaP_theo_mask, 'k--', sprintf('Yaklasik maskeleme siniri (%.1f dB)', DeltaP_theo_mask), 'LabelVerticalAlignment', 'bottom', 'FontSize', 8);
yline(0.90, 'k:', 'Pd = 0.90', 'LabelHorizontalAlignment', 'left', 'FontSize', 8);
grid on; ylim([0 1.05]); xlim([0 50]);
xlabel(sprintf('Guclu - Zayif Guc Farki \\DeltaP (dB) [SNR_{guclu} = %d dB]', SNR_strong_in)); ylabel('Zayif Hedef P_d');
title(sprintf('3) Asama A: Menzil Konumlari (Esit P_{fa} = %.1e)', Pfa_op), 'Color', 'k', 'FontSize', 10);
legend(h3, {'Hamming - CFAR Disi (10 hucre)', 'Dikdortgen - CFAR Disi (10 hucre)', ...
        'Dikdortgen - Egitim Bolgesi (4 hucre)', 'Hamming - Egitim Bolgesi (4 hucre)', ...
        'Hamming - Koruma Bolgesi (2 hucre)', 'Noktali: zayif hedef YOKKEN tespit (hayalet)'}, ...
        'Location', 'southwest', 'TextColor', 'k', 'Color', 'w', 'FontSize', 7);
style_axes(ax);

% Panel 4: yakin Doppler
ax = nexttile; hold on;
h4 = gobjects(1,4);
h4(1) = plot(Delta_P_vec, Pd_dop_rect,   'b--s', 'LineWidth', 1.6, 'MarkerFaceColor', 'b', 'MarkerSize', 5);
h4(2) = plot(Delta_P_vec, Pd_dop_hamm1D, 'r-o',  'LineWidth', 1.8, 'MarkerFaceColor', 'r', 'MarkerSize', 5);
h4(3) = plot(Delta_P_vec, Pd_dop_hamm2D, 'g-p',  'LineWidth', 1.8, 'MarkerFaceColor', 'g', 'MarkerSize', 6);
plot(Delta_P_vec, G_dop_rect,   'b:', 'LineWidth', 1.0);
plot(Delta_P_vec, G_dop_hamm1D, 'r:', 'LineWidth', 1.0);
plot(Delta_P_vec, G_dop_hamm2D, 'g:', 'LineWidth', 1.0);
h4(4) = plot(nan, nan, ':', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.0);
yline(0.90, 'k:', 'Pd = 0.90', 'LabelHorizontalAlignment', 'left', 'FontSize', 8);
grid on; ylim([0 1.05]); xlim([0 50]);
xlabel('Guclu - Zayif Guc Farki \DeltaP (dB)'); ylabel('Zayif Hedef P_d');
title('4) Asama A: Ayni Menzil, Yakin Doppler (\DeltaV = 5 Hucre)', 'Color', 'k', 'FontSize', 10);
legend(h4, {'Penceresiz', 'Hamming (yalniz Menzil)', 'Hamming (Menzil+Doppler)', ...
    'Noktali: zayif hedef YOKKEN tespit (hayalet)'}, 'Location', 'southwest', 'TextColor', 'k', 'Color', 'w', 'FontSize', 8);
style_axes(ax);

%% GRAFIK 2: ASAMA B - SABIT NOKTA
fig2 = figure('Color', 'w', 'Position', [60, 60, 1200, 720]);
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
BinU = [cfgs(idxU).Bh];  BinG = [cfgs(idxG).Bh];
Pfa_plot = Pfa_c;
Pfa_plot(zero_fa_flag) = Pfa_upper_zero;
Pfa_cal_plot = Pfa_c_cal;
Pfa_cal_plot(zero_fa_cal_flag) = Pfa_upper_zero;

ax = nexttile;
hU = plot(BinU, Pd_c(idxU), 'b-o', 'LineWidth', 1.6); hold on;
hG = plot(BinG, Pd_c(idxG), 'r-s', 'LineWidth', 1.6);
hUc = plot(BinU, Pd_c_cal(idxU), 'b:d', 'LineWidth', 1.5);
hGc = plot(BinG, Pd_c_cal(idxG), 'r:^', 'LineWidth', 1.5);
hF = plot([min([BinU BinG])-0.5, max([BinU BinG])+0.5], [Pd_flt_ref Pd_flt_ref], 'k--', 'LineWidth', 1.2);
plot([cfgs(okU).Bh], Pd_c(okU), 'bo', 'MarkerFaceColor', 'b', 'LineStyle', 'none');
plot([cfgs(okG).Bh], Pd_c(okG), 'rs', 'MarkerFaceColor', 'r', 'LineStyle', 'none');
grid on; ylim([0 1.05]); xlim([min([BinU BinG])-0.5, max([BinU BinG])+0.5]);
xlabel('Giris bit genisligi B_{h}'); ylabel(sprintf('Zayif Hedef P_d (DeltaP = %d dB)', Delta_P_fxp));
title('a) P_d: ortak alpha ve format-ozel esit-Pfa calibration', 'Color', 'k', 'FontSize', 9);
legend([hU hG hUc hGc hF], {'U - ortak alpha', 'G - ortak alpha', ...
    'U - esit-Pfa kalibrasyonu', 'G - esit-Pfa kalibrasyonu', 'Kayan nokta referans'}, ...
    'Location', 'southeast', 'TextColor', 'k', 'Color', 'w', 'FontSize', 7.5);
style_axes(ax);

ax = nexttile;
hU = semilogy(BinU, Pfa_plot(idxU), 'b-o', 'LineWidth', 1.8); hold on;
hG = semilogy(BinG, Pfa_plot(idxG), 'r-s', 'LineWidth', 1.8);
zAll = find(zero_fa_flag);
if ~isempty(zAll), hZ = semilogy([cfgs(zAll).Bh], Pfa_plot(zAll), 'kv', 'MarkerFaceColor', 'y', 'MarkerSize', 8); else, hZ = []; end
yline(Pfa_op, 'k-', 'Hedef P_{fa}', 'LabelHorizontalAlignment', 'left', 'FontSize', 8);
yline(0.5*Pfa_op, 'k:'); yline(2*Pfa_op, 'k:');
yline(Pfa_upper_zero, 'm--', sprintf('0 gozlem 95%% ust sinir (3/N_{cut}=%.1e)', Pfa_upper_zero), 'LabelHorizontalAlignment', 'right', 'FontSize', 7.5);
grid on; ylim([1e-6 1]); xlim([min([BinU BinG])-0.5, max([BinU BinG])+0.5]);
xlabel('Giris bit genisligi B_h'); ylabel('Sabit noktada olculen P_{fa}');
title('b) Ortak alpha ile sabit nokta P_{fa}', 'Color', 'k', 'FontSize', 9);
if ~isempty(hZ)
    legend([hU hG hZ], {'Aile U', 'Aile G', '0 Gozlem (nominal 3/N_{cut} ust siniri)'}, 'Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'FontSize', 7.5);
else
    legend([hU hG], {'Aile U', 'Aile G'}, 'Location', 'northeast', 'TextColor', 'k', 'Color', 'w', 'FontSize', 8);
end
style_axes(ax);

ax = nexttile;
hUcal = loglog(ratio_c(idxU), Pfa_cal_plot(idxU), 'b-o', 'LineWidth', 1.6); hold on;
hGcal = loglog(ratio_c(idxG), Pfa_cal_plot(idxG), 'r-s', 'LineWidth', 1.6);
yline(Pfa_op, 'k-', 'Hedef P_{fa}', 'LabelHorizontalAlignment', 'left', 'FontSize', 8);
yline(0.5*Pfa_op, 'k:'); yline(2*Pfa_op, 'k:');
grid on; ylim([1e-6 1]);
xlabel('LSB_Z / sigma_{gurultu}'); ylabel('Validation P_{fa} - format-ozel alpha');
title('c) Kalibrasyon sonrasi P_{fa}', 'Color', 'k', 'FontSize', 9);
legend([hUcal hGcal], {'Aile U', 'Aile G'}, 'Location', 'southwest', 'TextColor', 'k', 'Color', 'w', 'FontSize', 8);
style_axes(ax);

ax = nexttile;
yyaxis left;
plot(ratio_c(idxU), 100*DZ_c(idxU), 'b-o', 'LineWidth', 1.8); hold on;
plot(ratio_c(idxG), 100*DZ_c(idxG), 'r-s', 'LineWidth', 1.8);
ylabel('Olu bolge orani (% hucre, P = 0)'); ylim([0 105]);
ax.YColor = [0 0.3 0.8];
yyaxis right;
plot(ratio_c(idxU), rel_bg_nz(idxU), 'm--^', 'LineWidth', 1.5);
plot(ratio_c(idxG), rel_bg_nz(idxG), 'c-.v', 'LineWidth', 1.5, 'MarkerSize', 5);
set(ax, 'YScale', 'log');
ylabel('Sifir olmayan hucre ort. gucu / P_{bg,ref}');
ax.YColor = [0.75 0 0.5];
set(ax, 'XScale', 'log');
grid on; xlabel('LSB_{Z} / \sigma_{gurultu}');
title('d) Olu bolge orani ve kalan arka plan hucrelerinin goreli gucu', 'Color', 'k', 'FontSize', 9);
legend({'U: olu hucre', 'G: olu hucre', 'U: sifir olmayan goreli guc', 'G: sifir olmayan goreli guc'}, ...
    'Location', 'southwest', 'TextColor', 'k', 'Color', 'w', 'FontSize', 7);
style_axes(ax);

%% GRAFIK 3: ANA SABIT NOKTA MATRISI
fig3 = figure('Color','w','Position',[80,80,1280,800]);
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
Pfa6_raw_plot = Pfa6_raw; Pfa6_raw_plot(Pfa6_raw_plot==0) = PfaUpper6;
Pfa6_cal_plot = Pfa6_cal; Pfa6_cal_plot(Pfa6_cal_plot==0) = PfaUpper6;

% (a) Ortak alpha ile P_d
ax = nexttile; hold on; grid on;
selBits6 = [8 12 20];
cols6 = lines(2*numel(selBits6));
hA = gobjects(1, 2*numel(selBits6) + 2);
leg6 = cell(1, 2*numel(selBits6) + 2); kk = 0;
for wi = 1:2
    for sb = 1:numel(selBits6)
        bi = find(Bits6==selBits6(sb),1);
        kk = kk + 1;
        y = reshape(Pd6_raw(wi,bi,:),1,[]);
        lo = y - reshape(Pd6_raw_lo(wi,bi,:),1,[]);
        hi = reshape(Pd6_raw_hi(wi,bi,:),1,[]) - y;
        hA(kk) = errorbar(DeltaP6,y,lo,hi,'-o','Color',cols6(kk,:), 'LineWidth',1.5,'MarkerSize',5,'CapSize',4);
        leg6{kk} = sprintf('%s, %d bit',WinNames6{wi},selBits6(sb));
    end
end
lsty = {'k--','k-.'};
for wi = 1:2
    kk = kk + 1;
    hA(kk) = plot(DeltaP6, reshape(Pd6_float(wi,1,:),1,[]), lsty{wi}, 'LineWidth', 2);
    leg6{kk} = sprintf('%s, KAYAN NOKTA', WinNames6{wi});
end
yline(0.90,'k:','P_d = 0.90','Color','k');
ylim([0 1.05]); xlim([min(DeltaP6) max(DeltaP6)]);
xlabel('Guclu - zayif hedef guc farki \DeltaP (dB)','Color','k');
ylabel('Zayif hedef P_d (ortak alpha)','Color','k');
title('a) Sabit nokta P_d ve kayan nokta referansi','Color','k','FontSize',10);
legend(hA, leg6,'Location','southwest','FontSize',7,'TextColor','k','Color','w');
style_axes(ax);

% (b) Ortak alpha ile Pfa
ax = nexttile;
semilogy(Bits6,Pfa6_raw_plot(1,:),'-o','LineWidth',1.7,'MarkerSize',6); hold on; grid on;
semilogy(Bits6,Pfa6_raw_plot(2,:),'-s','LineWidth',1.7,'MarkerSize',6);
set(ax, 'YScale', 'log');
yline(Pfa_op,'k-','Hedef P_{fa}','Color','k','LabelHorizontalAlignment','left');
yline(PfaUpper6,'m--','0 gozlem 3/N ust siniri','LabelHorizontalAlignment','right');
xlim([min(Bits6)-0.5 max(Bits6)+0.5]); ylim([1e-6 1]);
xlabel('Toplam bit genisligi','Color','k'); ylabel('Validation P_{fa} (ortak alpha)','Color','k');
title('b) Ortak alpha ile nicemleme kaynakli P_{fa} sapmasi','Color','k','FontSize',10);
legend({'Dikdortgen','Hamming - menzil'},'Location','northeast','TextColor','k','Color','w');
style_axes(ax);

% (c) Format-ozel alpha ile P_d
ax = nexttile; hold on; grid on;
hC = gobjects(1,4);
for wi = 1:2
    y = reshape(Pd6_cal(wi,:,end),1,[]);
    lo = y - reshape(Pd6_cal_lo(wi,:,end),1,[]);
    hi = reshape(Pd6_cal_hi(wi,:,end),1,[]) - y;
    hC(wi) = errorbar(Bits6,y,lo,hi,'--o','LineWidth',1.6,'MarkerSize',5,'CapSize',4);
    hC(2+wi) = plot([min(Bits6)-0.5 max(Bits6)+0.5], Pd6_float(wi,1,end)*[1 1], lsty{wi}, 'LineWidth', 1.5);
end
yline(0.90,'k:','P_d = 0.90','Color','k'); ylim([0 1.05]);
xlim([min(Bits6)-0.5 max(Bits6)+0.5]);
xlabel('Toplam bit genisligi','Color','k');
ylabel(sprintf('P_d (\\DeltaP = %d dB, format-ozel alpha)',DeltaP6(end)),'Color','k');
title('c) Zor kosulda bit genisligi (bos nokta = kalibre edilemedi)','Color','k','FontSize',10);
legend(hC, {'Dikdortgen - sabit','Hamming - sabit','Dikdortgen - KAYAN','Hamming - KAYAN'}, ...
    'Location','northwest','TextColor','k','Color','w','FontSize',8);
style_axes(ax);

% (d) Format-ozel alpha ile Pfa
ax = nexttile;
semilogy(Bits6,Pfa6_cal_plot(1,:),'--o','LineWidth',1.7,'MarkerSize',6); hold on; grid on;
semilogy(Bits6,Pfa6_cal_plot(2,:),'--s','LineWidth',1.7,'MarkerSize',6);
set(ax, 'YScale', 'log');
yline(Pfa_op,'k-','Hedef P_{fa}','Color','k','LabelHorizontalAlignment','left');
yline(0.5*Pfa_op,'k:'); yline(2*Pfa_op,'k:');
yline(PfaUpper6,'m--','0 gozlem 3/N ust siniri','LabelHorizontalAlignment','right');
xlim([min(Bits6)-0.5 max(Bits6)+0.5]); ylim([1e-6 1]);
xlabel('Toplam bit genisligi','Color','k'); ylabel('Validation P_{fa} (format-ozel alpha)','Color','k');
title('d) Kalibrasyon sonrasi bagimsiz validation P_{fa}','Color','k','FontSize',10);
legend({'Dikdortgen','Hamming - menzil'},'Location','northeast','TextColor','k','Color','w');
style_axes(ax);

%% GRAFIK 4: NICEMLEME KAYBI
fig4 = figure('Color','w','Position',[100,100,1350,430]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

% (a) Kayan nokta Pd
ax = nexttile; hold on; grid on;
hR = gobjects(1,4);
mk = {'-o','-s'}; mk0 = {':o',':s'};
for wi = 1:2
    hR(wi)   = plot(DeltaP6, reshape(Pd6_float(wi,1,:),1,[]), mk{wi}, 'LineWidth', 1.8, 'MarkerSize', 6);
    hR(2+wi) = plot(DeltaP6, reshape(PweakROI0_6_float(wi,1,:),1,[]), mk0{wi}, 'LineWidth', 1.2, 'MarkerSize', 5);
end
ylim([0 1.05]); xlim([min(DeltaP6) max(DeltaP6)]);
xlabel('\DeltaP (dB)'); ylabel('Kayan nokta P_d');
title('a) Nicemleme YOK: termal gurultu + yan lob', 'Color','k','FontSize',10);
legend(hR, {'Dikdortgen - P_d','Hamming - P_d','Dikdortgen - hayalet','Hamming - hayalet'}, ...
    'Location','southwest','TextColor','k','Color','w','FontSize',8);
style_axes(ax);

% (b)-(c) Pencereye gore nicemleme kaybi
for wi = 1:2
    ax = nexttile; hold on; grid on;
    cmap = lines(nD6);
    hL = gobjects(1, nD6 + 1);
    for di = 1:nD6
        hL(di) = plot(Bits6, reshape(DeltaPd6_cal(wi,:,di),1,[]), '-o', 'Color', cmap(di,:), ...
            'LineWidth', 1.6, 'MarkerSize', 5);
    end
    yline(0, 'k-');
    na = find(~Alpha6_possible(wi,:));
    hL(end) = plot(nan, nan, 'kx', 'MarkerSize', 10, 'LineWidth', 2);
    if ~isempty(na)
        plot(Bits6(na), -0.08*ones(1,numel(na)), 'kx', 'MarkerSize', 10, 'LineWidth', 2);
    end
    ylim([-0.15 1.05]); xlim([min(Bits6)-0.5 max(Bits6)+0.5]);
    xlabel('Toplam bit genisligi'); ylabel('\DeltaP_d = P_d(kayan) - P_d(sabit, esit P_{fa})');
    title(sprintf('%c) %s: yalniz nicemleme kaybi', 'a'+wi, WinNames6{wi}), 'Color','k','FontSize',10);
    legend(hL, [arrayfun(@(d) sprintf('\\DeltaP = %d dB', d), DeltaP6, 'UniformOutput', false), ...
        {'x: kalibre edilemedi'}], 'Location','northeast','TextColor','k','Color','w','FontSize',8);
    style_axes(ax);
end

%% KAYIT
exportgraphics(fig1, 'sekil1_cfar_ve_konum.png', 'Resolution', 200);
exportgraphics(fig2, 'sekil2_sabit_nokta.png', 'Resolution', 200);
exportgraphics(fig3, 'sekil3_ana_matris.png', 'Resolution', 200);
exportgraphics(fig4, 'sekil4_nicemleme_kaybi.png', 'Resolution', 200);
save('sonuclar.mat', 'sim_cfg', 'cl', 'cfgs', ...
     'Pd_c', 'Pd_ci_lo', 'Pd_ci_hi', 'Pd_c_cal', 'Pd_cal_ci_lo', 'Pd_cal_ci_hi', 'SQNR_c', ...
     'Pfa_c', 'Pfa_ci_lo', 'Pfa_ci_hi', 'Pfa_c_cal', 'Pfa_cal_ci_lo', 'Pfa_cal_ci_hi', ...
     'alpha_fxp_cal', 'alpha_fxp_possible', 'alpha_fxp_train_pfa', ...
     'alpha_cal_rect', 'alpha_cal_hamm1D', 'alpha_cal_hamm2D', 'alpha_q8_shared', ...
     'alpha_theo_op', 'Pfa_op', 'DeltaP_theo_mask', 'N_train', 'Tr', 'Td', 'Gr', 'Gd', ...
     'n_cal_noise', 'n_cal_fxp', 'n_val_fxp', 'Pfa_upper_zero_float', 'Pfa_upper_zero', ...
     'DZ_c', 'Pbg_all_c', 'Pbg_nz_c', 'ratio_c', 'SAT_c', 'cfar_ok', 'cfar_ok_cal', ...
     'Pd_flt_ref', 'Pd_flt_ref_lo', 'Pd_flt_ref_hi', ...
     'Delta_P_vec', 'PdA_CI', 'Pd_guard_hamm', 'Pd_train_rect', 'Pd_train_hamm1D', ...
     'Pd_out_rect', 'Pd_out_hamm1D', 'Pd_dop_rect', 'Pd_dop_hamm1D', 'Pd_dop_hamm2D', ...
     'G_guard_hamm', 'G_train_rect', 'G_train_hamm1D', 'G_out_rect', 'G_out_hamm1D', ...
     'G_dop_rect', 'G_dop_hamm1D', 'G_dop_hamm2D', ...
     'Pfa_targets', 'Pfa_meas_rect', 'Pfa_meas_hamm', 'Pfa_meas_hamm2D', ...
     'Pfa_ci_rect', 'Pfa_ci_hamm', 'Pfa_ci_hamm2D', ...
     'Bits6', 'DeltaP6', 'N_mc6', 'WinNames6', 'Pd6_raw', 'Pd6_raw_lo', 'Pd6_raw_hi', 'Pd6_cal', 'Pd6_cal_lo', 'Pd6_cal_hi', ...
     'Pd6_float', 'Pd6_float_lo', 'Pd6_float_hi', 'PweakROI0_6_float', 'DeltaPd6_cal', 'DeltaPd6_raw', ...
     'Pfa6_raw', 'Pfa6_raw_lo', 'Pfa6_raw_hi', 'Pfa6_cal', 'Pfa6_cal_lo', 'Pfa6_cal_hi', ...
     'Alpha6_shared', 'Alpha6_cal', 'Alpha6_possible', 'Pfa6_within_factor2', 'PfaUpper6', 'SQNR6_dB', ...
     'PweakROI0_6', 'PweakROI0_6_cal', 'Pd6_incremental', 'Pd6_incremental_cal', 'DZ6', 'SAT6');
fprintf('\nSekiller ve sonuclar kaydedildi.\n');

%% YEREL FONKSIYONLAR
function hit_flag = hit_resolved(P, th, r_weak, v_weak, r_strong, v_strong)
    % Guclu hedef cevresini maskeleyip zayif hedef penceresinde tespit arar
    det_map = (P > th);
    rs = max(1,r_strong-1):min(size(P,1),r_strong+1);
    vs = max(1,v_strong-1):min(size(P,2),v_strong+1);
    det_map(rs,vs) = false;
    rw = max(1,r_weak-1):min(size(P,1),r_weak+1);
    vw = max(1,v_weak-1):min(size(P,2),v_weak+1);
    weak_region = det_map(rw,vw);
    hit_flag = any(weak_region(:));
end

function [P, th, sat_max, Pn, lsb_P, clip_th, Z] = fxp_chain(H, cf, Wq, cl, ksum, inv_q, alpha_q)
    % Blok seviyesinde sabit nokta isleme zinciri
    [Nn, Mm] = size(H);
    Ntrain = sum(ksum(:));

    [Hq, s_h] = quant_cplx(H, cf.Bh, cl.H);
    Hwin_raw = Hq .* Wq;
    [Hwin, s_mul] = quant_cplx(Hwin_raw, cf.Bh, cl.H);
    [s1, s_1] = quant_cplx(ifft(Hwin, Nn, 1) * sqrt(Nn), cf.Bs, cl.s1);
    [Z, s_z]  = quant_cplx(fft(s1, Mm, 2) / sqrt(Mm), cf.Bz, cl.Z);

    lsb_Z = cl.Z / 2^(cf.Bz-1);
    lsb_P = lsb_Z^2;
    Bp = 2*cf.Bz + 1;
    clip_P = min(2*(cl.Z^2), (2^Bp-1)*lsb_P);
    P_raw = real(Z).^2 + imag(Z).^2;
    s_p = mean(P_raw(:) >= clip_P - 0.5*lsb_P);
    P = max(0, min(clip_P-lsb_P, round(P_raw/lsb_P)*lsb_P));

    Bsum = Bp + ceil(log2(Ntrain));
    clip_sum_repr = (2^Bsum-1)*lsb_P;
    clip_sum_phys = Ntrain*(clip_P-lsb_P);
    clip_sum = min(clip_sum_repr, clip_sum_phys);
    sum_raw = conv2(P, ksum, 'same');
    s_sum = mean(sum_raw(:) > clip_sum + 0.5*lsb_P);
    sum_train = max(0, min(clip_sum, round(sum_raw/lsb_P)*lsb_P));

    Pn_raw = sum_train * inv_q;
    Pn = max(0, min(clip_P-lsb_P, floor(Pn_raw/lsb_P)*lsb_P));

    Bth = Bp + 9;
    clip_th = (2^Bth-1)*lsb_P;
    [th, s_th] = fxp_threshold(Pn, alpha_q, lsb_P, clip_th);

    sat_max = max([s_h, s_mul, s_1, s_z, s_p, s_sum, s_th]);
end

function [th, sat_rate] = fxp_threshold(Pn, alpha_q, lsb_P, clip_th)
    th_raw = Pn .* alpha_q;
    sat_rate = mean(th_raw(:) >= clip_th - 0.5*lsb_P);
    th_q = round(th_raw/lsb_P)*lsb_P;
    th = max(0, min(clip_th-lsb_P, th_q));
end

function [alpha_q, possible, pfa_at_cal] = calibrate_fxp_alpha(Pvals, Pnvals, lsb_P, clip_th, target_pfa)
    pfa_at_zero = mean(Pvals > fxp_threshold(Pnvals, 0, lsb_P, clip_th));
    if pfa_at_zero < target_pfa
        alpha_q = NaN; possible = false; pfa_at_cal = pfa_at_zero; return;
    end
    lo = 0; hi = 65535;
    pfa_hi = mean(Pvals > fxp_threshold(Pnvals, hi/256, lsb_P, clip_th));
    if pfa_hi > target_pfa
        alpha_q = NaN; possible = false; pfa_at_cal = pfa_hi; return;
    end
    while lo < hi
        mid = floor((lo+hi)/2);
        pfa_mid = mean(Pvals > fxp_threshold(Pnvals, mid/256, lsb_P, clip_th));
        if pfa_mid > target_pfa
            lo = mid + 1;
        else
            hi = mid;
        end
    end
    alpha_q = lo/256;
    pfa_at_cal = mean(Pvals > fxp_threshold(Pnvals, alpha_q, lsb_P, clip_th));
    possible = true;
end

function alpha = empirical_alpha(ratios, target_pfa)
    x = ratios(:);
    x = sort(x(isfinite(x)));
    if isempty(x)
        error('Alpha kalibrasyonu icin sonlu CUT/ortalama orani bulunamadi.');
    end
    idx = ceil((1-target_pfa)*numel(x));
    idx = max(1, min(numel(x), idx));
    alpha = x(idx);
end

function [y, sat] = quant_cplx(x, B, clipv)
    s  = clipv / 2^(B-1);
    ur = real(x); ui = imag(x);
    y  = quant_real(ur, B, clipv) + 1j*quant_real(ui, B, clipv);
    sat = mean([abs(ur(:)) >= clipv - s/2; abs(ui(:)) >= clipv - s/2]);
end

function y = quant_real(u, B, clipv)
    s = clipv / 2^(B-1);
    y = max(-clipv, min(clipv - s, round(u/s)*s));
end

function [ci_lo, ci_hi] = binom_ci95(k, n)
    % Wilson %95 guven araligi
    z = 1.95996;
    p_hat = k / n;
    denom = 1 + (z^2)/n;
    center = (p_hat + (z^2)/(2*n)) / denom;
    half_w = (z * sqrt((p_hat*(1 - p_hat)/n) + (z^2)/(4*(n^2)))) / denom;
    ci_lo = max(0, center - half_w);
    ci_hi = min(1, center + half_w);
end

function s = num_or_na(x)
    if isfinite(x), s = sprintf('%.2f', x); else, s = 'N/A'; end
end

function style_axes(ax)
    set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', [0.3 0.3 0.3]);
end
