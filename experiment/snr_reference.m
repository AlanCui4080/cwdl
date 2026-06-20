#!/usr/bin/octave -qf
% 完整脚本：读取 CW 音频，绘频谱图，自动估计 SNR（无额外依赖）

% ===== 1. 加载必要包 =====
try
    pkg load signal
catch
    fprintf('未安装 signal 包，请运行以下命令后重试：\n');
    fprintf('pkg install -forge signal\n');
    fprintf('pkg load signal\n');
    error('缺少 signal 包');
end

% ===== 2. 读取音频 =====
filename = "John316MorseCode.wav";          % 请修改为实际文件名
[y, fs] = audioread(filename);
fprintf('采样率: %d Hz, 长度: %.2f 秒\n', fs, length(y)/fs);



% ===== 6. SNR 自动估计（直接计算，无函数）=====
fc = 700;                           % CW 中心频率 (Hz)
bw = 200;                           % 带宽 (Hz)

% 6.1 带通滤波
[b, a] = butter(6, [fc-bw/2, fc+bw/2] / (fs/2));
y_filt = filter(b, a, y);

% 6.2 包络检测
envelope = abs(y_filt);
[b_lp, a_lp] = butter(2, 50 / (fs/2));   % 50 Hz 低通
envelope = filter(b_lp, a_lp, envelope);

% 6.3 阈值分割：2倍中位数作为初步阈值
thresh = 2 * median(envelope);
mask_on  = (envelope >= thresh);
mask_off = ~mask_on;

% 6.4 若分割极不平衡，改用十分位数均值（不依赖 prctile 函数）
if sum(mask_on) < 10 || sum(mask_off) < 10
    sorted_env = sort(envelope);
    n = length(sorted_env);
    p10 = sorted_env(round(0.10 * n));   % 第10百分位
    p90 = sorted_env(round(0.90 * n));   % 第90百分位
    thresh = 0.5 * (p10 + p90);
    mask_on  = (envelope >= thresh);
    mask_off = ~mask_on;
end

% 6.5 计算信号功率和噪声功率
p_total_on = mean(y(mask_on).^2);        % 含噪信号段总功率
p_noise    = mean(y(mask_off).^2);       % 纯噪声段功率
p_signal   = p_total_on - p_noise;       % 信号功率

% 6.6 得到 SNR (dB)
if p_signal <= 0
    snr_db = -Inf;
else
    snr_db = 10 * log10(p_signal / p_noise);
end

% ===== 7. 输出结果 =====
fprintf('自动估计信噪比 (SNR) = %.2f dB\n', snr_db);
