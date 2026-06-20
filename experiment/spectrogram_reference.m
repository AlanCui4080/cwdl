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
end

% 文件名
filename = "morse_73.wav";

% 读取音频
try
    [y, fs] = audioread(filename);
catch
    fprintf('无法读取文件 "%s"，请检查路径。\n', filename);
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
