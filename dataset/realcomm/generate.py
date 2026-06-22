import random
import string

LETTERS = string.ascii_uppercase
DIGITS = string.digits
ALNUM = LETTERS + DIGITS

# 大洲缩写 + DX + TEST 作为比赛名
CONTEST_NAMES = ["NA", "SA", "EU", "AS", "AF", "OC", "AN", "DX", "TEST"]

NUM_LINES = 1000


def rand_callsign():
    """1-2 个随机字母 + 一个数字 + 2-3 个随机字母"""
    prefix = ''.join(random.choices(LETTERS, k=random.randint(1, 2)))
    digit = random.choice(DIGITS)
    suffix = ''.join(random.choices(LETTERS, k=random.randint(2, 3)))
    return prefix + digit + suffix


def gen_serial():
    """0-999 补零至 3 位（如 001, 023, 123）"""
    serial = f"{random.randint(0, 999):03d}"
    return serial


def corrupt(token):
    """随机损坏 1-2 个字符"""
    chars = list(token)
    n_err = random.randint(1, min(2, len(chars)))
    for p in random.sample(range(len(chars)), n_err):
        chars[p] = random.choice(ALNUM)
    return ''.join(chars)


def maybe_error(token):
    """2% 概率出错：损坏 1-2 字符 + [DEL] + 正确序列"""
    if random.random() >= 0.02:
        return token
    corrupted = corrupt(token)
    return f"{corrupted} [DEL] {token}"


def longtail_count():
    """1-5 的长尾分布：1 遍 70%，2-5 递减"""
    if random.random() < 0.7:
        return 1
    # 剩余 30% 在 2-5 之间递减分配
    weights = [12, 9, 6, 3]
    r = random.random() * sum(weights)
    acc = 0
    for i, w in enumerate(weights):
        acc += w
        if r < acc:
            return 2 + i
    return 5


def gen_qso():
    contest = random.choice(CONTEST_NAMES)
    call_a = rand_callsign()
    call_b = rand_callsign()

    # 1) 1-2 个 CQ + 比赛名 + 呼号A，原样重复 1-3 遍，间隔 20-100 空格
    def cq_segment():
        return ' '.join(["CQ"] * random.randint(1, 2) + [contest, maybe_error(call_a)])

    part1 = (' ' * random.randint(20, 100)).join(
        cq_segment() for _ in range(random.randint(1, 3)))

    # 2) 呼号B，原样重复 1-5 遍（长尾），间隔 10-30 空格
    n_b = longtail_count()
    part2 = (' ' * random.randint(10, 30)).join(
        maybe_error(call_b) for _ in range(n_b))

    # 3) 呼号B 5NN [NR 序号] [50%: K]
    #    2% 抄错 / 2% 听不清
    has_nr = True
    has_k3 = random.random() < 0.50
    serial3 = gen_serial() if has_nr else None
    serial4 = gen_serial() if has_nr else None

    def build_exchange(call, with_k):
        parts = [call, "5NN"]
        if has_nr:
            parts += ["NR", maybe_error(serial3)]
        if with_k:
            parts.append("K")
        return ' '.join(parts)

    roll = random.random()
    if roll < 0.02:
        # 抄错：错呼号 + 5NN [NR] + QRZ + 正确呼号x2 + 间隔 + 重发上一句
        wrong_b = corrupt(call_b)
        first = ' '.join([wrong_b, "5NN"] + (["NR", serial3] if has_nr else []))
        qrz = f"QRZ {call_b} {call_b}"
        resend = build_exchange(maybe_error(call_b), has_k3)
        part3 = first + ' ' + qrz + ' ' * random.randint(5, 20) + resend
    elif roll < 0.04:
        # 听不清：QRZ? + 间隔 + 呼号1-2遍 + 间隔 + 重发上一句
        calls = ' '.join([call_b] * random.randint(1, 2))
        resend = build_exchange(maybe_error(call_b), has_k3)
        part3 = "QRZ?" + ' ' * random.randint(5, 20) + calls + ' ' * random.randint(5, 20) + resend
    else:
        part3 = build_exchange(maybe_error(call_b), has_k3)

    # 4) TU 呼号A 5NN [NR 序号] [20%: BK]
    parts4 = ["TU", maybe_error(call_a), "5NN"]
    if has_nr:
        parts4 += ["NR", maybe_error(serial4)]
    if random.random() < 0.2:
        parts4.append("BK")
    part4 = ' '.join(parts4)

    # 段间间隔：part1→part2 10-30空格，part2→part3 和 part3→part4 5-20空格
    result = (part1
              + ' ' * random.randint(10, 30) + part2
              + ' ' * random.randint(5, 20) + part3
              + ' ' * random.randint(5, 20) + part4)

    # 5% 概率尾部追加 2-3 个间隔两空格的 E
    if random.random() < 0.05:
        result += ' ' + '  '.join(['E'] * random.randint(2, 3))

    return result


with open("realcomm.txt", "w", encoding="utf-8") as f:
    for _ in range(NUM_LINES):
        f.write(gen_qso() + "\n")

print(f"Generated {NUM_LINES} real contest QSO sequences to realcomm.txt")
