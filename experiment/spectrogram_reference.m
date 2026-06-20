#!/usr/bin/octave -qf
% 文件名可以任意，例如 cw_snr.m （不要与内部函数同名）
% 功能：读取 CW 音频，绘频谱图，自动估计 SNR

% 检查所需包
try
    pkg load signal
    pkg load statistics
catch
    fprintf('缺少必要的包，请运行以下命令后重试：\n');
    fprintf('pkg install -forge signal statistics\n');
    fprintf('pkg load signal statistics\n');
    exit(1);
end

% 文件名
filename = "morse_73.wav";

% 读取音频
try
    [y, fs] = audioread(filename);
catch
    fprintf('无法读取文件 "%s"，请检查路径。\n', filename);
    exit(1);
end

% ===== 参数设置 =====
winlen = 480;               % 窗口长度（样本数）
overlap = winlen * 0.707;   % 重叠样本数
nfft = 4800;                % FFT 点数
f_max = 4000;               % 显示的最高频率

% ===== 计算短时傅里叶变换 (STFT) =====
[S, f, t] = specgram(y, nfft, fs, hanning(winlen), overlap);
S_dB = 20 * log10(abs(S) + eps);

% ===== 绘制频谱图 =====
figure('Name', '时频图 (Spectrogram)');
imagesc(t, f, S_dB);
axis xy;                    % 低频在下方
xlabel('时间 (s)');
ylabel('频率 (Hz)');
title('CW 信号频谱图 (幅度谱, dB)');
colorbar;
ylim([0 f_max]);

% ===== 调用 SNR 自动估计 =====
% 假设 CW 中心频率 700 Hz，带宽 200 Hz（可手动调整）
fc = 700;
bw = 200;
snr_est = snr_cw_auto(y, fs, fc, bw);
fprintf('自动估计信噪比 (SNR) = %.2f dB\n', snr_est);

% ============================================
%  函数定义（放在脚本末尾）
% ============================================
function snr_db = snr_cw_auto(y, fs, fc, bw)
    % 自动估计 CW 信号的信噪比
    % 输入：y - 音频信号，fs - 采样率，fc - 中心频率，bw - 带宽
    % 输出：snr_db (dB)

    % 带通滤波器
    [b, a] = butter(6, [fc-bw/2, fc+bw/2] / (fs/2));
    y_filt = filter(b, a, y);

    % 包络检测
    envelope = abs(y_filt);
    [b_lp, a_lp] = butter(2, 50/(fs/2));  % 50 Hz 低通
    envelope = filter(b_lp, a_lp, envelope);

    % 两均值聚类分离 on/off
    try
        [idx, centers] = kmeans(envelope(:), 2, 'Start', [min(envelope); max(envelope)]);
    catch
        % 若 kmeans 失败，采用简单的 Otsu 阈值
        thresh = graythresh(envelope);
        idx = (envelope > thresh * max(envelope)) + 1;
        centers = [mean(envelope(idx==1)); mean(envelope(idx==2))];
    end

    % 确保 centers(1) 为低幅段（噪声）
    if centers(1) > centers(2)
        idx = 3 - idx;
        centers = centers([2,1]);
    end

    mask_on = (idx == 2);
    mask_off = (idx == 1);

    p_total_on = mean(y(mask_on).^2);
    p_noise = mean(y(mask_off).^2);
    p_signal = p_total_on - p_noise;

    if p_signal <= 0
        snr_db = -Inf;
    else
        snr_db = 10 * log10(p_signal / p_noise);
    end
end
