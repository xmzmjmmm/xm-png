#!/usr/bin/env python3
import sys, os, zlib, struct, hashlib, glob
import numpy as np
from PIL import Image

MAGIC = b'XMF2'
GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _perm_seed(password, salt):
    return int.from_bytes(hashlib.sha256(password.encode('utf-8') + salt).digest()[:8], 'big')


def _keys(perm, total):
    z = (np.uint64(perm) + np.arange(total, dtype=np.uint64) * GOLDEN)
    z ^= z >> np.uint64(30)
    z *= np.uint64(0xBF58476D1CE4E5B9)
    z ^= z >> np.uint64(27)
    z *= np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    return z


def _order_top(perm, total, k):
    z = _keys(perm, total)
    k = min(k, total)
    if k < total:
        top = np.argpartition(z, k - 1)[:k]
        o = np.argsort(z[top], kind='stable')
    else:
        top = np.arange(total)
        o = np.argsort(z, kind='stable')
    return top[o].astype(np.int64)


def _expand(arr, need_units):
    h, w, c = arr.shape
    if h * w * c >= need_units:
        return arr
    scale = 1.0
    while h * scale * w * scale * c < need_units:
        scale *= 1.5
    nh, nw = int(np.ceil(h * scale)), int(np.ceil(w * scale))
    out = np.zeros((nh, nw, c), dtype=np.uint8)
    out[:h, :w] = arr
    for i in range((nw - w + w - 1) // w):
        x0 = w + i * w
        seg = arr[:, ::-1] if i % 2 == 0 else arr
        out[:h, x0:min(nw, x0 + w)] = seg[:, :min(w, nw - x0)]
    top = out[:h].copy()
    for i in range((nh - h + h - 1) // h):
        y0 = h + i * h
        src = top[::-1] if i % 2 == 0 else top
        out[y0:min(nh, y0 + h)] = src[:min(h, nh - y0)]
    rng = np.random.default_rng(7)
    mask = np.ones((nh, nw), dtype=bool)
    mask[:h, :w] = False
    noise = rng.integers(-2, 3, out.shape, dtype=np.int16)
    out[mask] = np.clip(out[mask].astype(np.int16) + noise[mask], 0, 255).astype(np.uint8)
    return out


def _mask(d):
    return np.uint8((1 << d) - 1)


def _pack_bits(bits, d):
    pad = (-len(bits)) % d
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    shifts = np.arange(d - 1, -1, -1, dtype=np.uint64)
    return (bits.reshape(-1, d).astype(np.uint64) << shifts).sum(axis=1, dtype=np.uint64).astype(np.uint8)


def _unpack_bits(vals, d, nbits):
    shifts = np.arange(d - 1, -1, -1, dtype=np.uint64)
    return ((vals.astype(np.uint64)[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.uint8).reshape(-1)[:nbits]


def _frame(payload, nbits):
    body = struct.pack('>IB', len(payload), nbits) + payload
    crc = zlib.crc32(body) & 0xFFFFFFFF
    hdr = MAGIC + struct.pack('>I', crc)
    frame_b = hdr + body
    return hdr + frame_b + hdr, frame_b


def embed(in_path, out_path, payload, password, nbits=1):
    if isinstance(payload, str):
        payload = payload.encode('utf-8')
    if nbits not in (1, 2, 3, 4):
        raise ValueError('位深只支持 1-4')
    arr = np.array(Image.open(in_path).convert('RGB'))
    salt = os.urandom(8)
    perm = _perm_seed(password, salt)
    full, frame_b = _frame(payload, nbits)
    bits = np.unpackbits(np.frombuffer(full, np.uint8))
    need_units = (len(bits) + nbits - 1) // nbits
    arr2 = _expand(arr, need_units + 64)
    h, w, c = arr2.shape
    flat = arr2.reshape(-1)
    sbits = np.unpackbits(np.frombuffer(salt, np.uint8))
    flat[-64:] = (flat[-64:] & np.uint8(0xFE)) | sbits
    m = flat.size - 64
    if m < need_units:
        raise ValueError('数据超出容量')
    idx = _order_top(perm, m, need_units)
    flat[idx] = (flat[idx] & (np.uint8(0xFF) ^ _mask(nbits))) | _pack_bits(bits, nbits)
    Image.fromarray(flat.reshape(h, w, c)).save(out_path, format='PNG', optimize=False)
    return len(payload), (h, w), nbits


def capacity(in_path, nbits=1):
    arr = Image.open(in_path).convert('RGB')
    n = arr.width * arr.height * 3 - 64
    return n * nbits // 8 - 29


_HDR_UNITS = 64


def extract(stego_path, password):
    flat = np.array(Image.open(stego_path).convert('RGB')).reshape(-1)
    salt = np.packbits((flat[-64:] & 1).astype(np.uint8)).tobytes()
    perm = _perm_seed(password, salt)
    m = flat.size - 64
    po = _order_top(perm, m, min(m, _HDR_UNITS))
    d = None
    for cand in (1, 2, 3, 4):
        k = (64 + cand - 1) // cand
        bits = _unpack_bits(flat[po[:k]] & _mask(cand), cand, 64)
        if np.packbits(bits).tobytes()[:4] == MAGIC:
            d = cand
            break
    if d is None:
        raise ValueError('无数据或密码错误')
    probe_units = min(m, (8 * 21 + d - 1) // d)
    po2 = po if probe_units <= len(po) else _order_top(perm, m, probe_units)
    probe_raw = np.packbits(_unpack_bits(flat[po2[:probe_units]] & _mask(d), d, 8 * 21)).tobytes()
    plen = struct.unpack('>I', probe_raw[16:20])[0]
    need_bits = 8 * (29 + plen)
    need_units = (need_bits + d - 1) // d
    if need_units > m:
        raise ValueError('数据超出容量')
    order = po if need_units <= len(po) else _order_top(perm, m, need_units)
    raw_bits = _unpack_bits(flat[order[:need_units]] & _mask(d), d, need_bits)
    raw = np.packbits(raw_bits).tobytes()
    frame_b = raw[8:21 + plen]
    if zlib.crc32(frame_b[8:]) & 0xFFFFFFFF != struct.unpack('>I', frame_b[4:8])[0]:
        raise ValueError('数据损坏或密码错误')
    plen2, nb = struct.unpack('>IB', frame_b[8:13])
    return frame_b[13:13 + plen2]


def _ask(prompt):
    return input(prompt).strip()


def _save_out(path, payload):
    try:
        print('内容:', payload.decode('utf-8'))
    except UnicodeDecodeError:
        dest = path + '.out'
        open(dest, 'wb').write(payload)
        print(f'二进制 -> {dest} ({len(payload)}B)')


def _embed_text():
    src, dst = _ask('原图路径: '), _ask('输出路径: ')
    data = _ask('要藏的内容: ')
    pw = _ask('主密码: ')
    d = int(_ask('位深 1-4 (默认1): ') or 1)
    n, size, nb = embed(src, dst, data, pw, d)
    print(f'已嵌入 {n}B -> {dst} 画布={size} 位深={nb}')


def _embed_file():
    src, dst = _ask('原图路径: '), _ask('输出路径: ')
    fp = _ask('文件路径: ')
    pw = _ask('主密码: ')
    d = int(_ask('位深 1-4 (默认1): ') or 1)
    n, size, nb = embed(src, dst, open(fp, 'rb').read(), pw, d)
    print(f'已嵌入 {n}B -> {dst} 画布={size} 位深={nb}')


def _extract():
    p = _ask('图片路径: ')
    _save_out(p, extract(p, _ask('主密码: ')))


def _scan():
    dirp, pw = _ask('目录: '), _ask('主密码: ')
    found = 0
    for f in sorted(glob.glob(os.path.join(dirp, '**', '*.png'), recursive=True)):
        try:
            _save_out(f, extract(f, pw))
            found += 1
        except Exception:
            pass
    print(f'共找到 {found} 张')


def main():
    if len(sys.argv) < 2:
        menu = '1 嵌入\n2 嵌入文件\n3 提取\n4 扫描\n0 退出\n'
        while True:
            print(menu)
            c = _ask('请选择: ')
            try:
                if c == '1':
                    _embed_text()
                elif c == '2':
                    _embed_file()
                elif c == '3':
                    _extract()
                elif c == '4':
                    _scan()
                elif c in ('0', 'q', 'exit'):
                    return
            except KeyboardInterrupt:
                return
            except Exception as e:
                print('出错:', e)
        return
    cmd, a = sys.argv[1], sys.argv[2:]
    try:
        if cmd in ('1', 'embed'):
            payload = open(a[2], 'rb').read() if os.path.isfile(a[2]) else a[2].encode('utf-8')
            n, size, nb = embed(a[0], a[1], payload, a[3], int(a[4]) if len(a) > 4 else 1)
            print(f'embedded {n}B -> {a[1]} canvas={size} bits={nb}')
        elif cmd in ('2', 'extract'):
            _save_out(a[0], extract(a[0], a[1]))
        elif cmd in ('3', 'embedf'):
            n, size, nb = embed(a[0], a[1], open(a[2], 'rb').read(), a[3], int(a[4]) if len(a) > 4 else 1)
            print(f'embedded file {n}B -> {a[1]} canvas={size} bits={nb}')
        elif cmd in ('5', 'cap'):
            for d in (1, 2, 3, 4):
                print(f'{d}bit: {capacity(a[0], d)}B')
        elif cmd in ('4', 'scan'):
            dirp, pw = a[0], a[1]
            found = 0
            for f in sorted(glob.glob(os.path.join(dirp, '**', '*.png'), recursive=True)):
                try:
                    _save_out(f, extract(f, pw))
                    found += 1
                except Exception:
                    pass
            print(f'共找到 {found} 张')
        else:
            print('用法: xm.py [1|2|3|4|5] ...')
    except Exception as e:
        print('出错:', e)


if __name__ == '__main__':
    main()

