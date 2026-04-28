import argparse, cv2, torch, numpy as np
from pathlib import Path
from moge.model.v2 import MoGeModel

def apply_clahe(img, clip=2.0, tile=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_Lab2BGR)

def apply_denoise(img, h=4):
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

def apply_sharpen(img, amount=1.0):
    blurred = cv2.GaussianBlur(img, (0,0), 3)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
'''
def upscale_image(img, model_name='realesrgan', outscale=2):
    if model_name == 'realesrgan':
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        # Скачайте модель заранее: RealESRGAN_x4plus.pth
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path='weights/RealESRGAN_x4plus.pth',
            model=model,
            tile=0, tile_pad=10, pre_pad=0, half=False
        )
        # Real-ESRGAN всегда увеличивает в 4 раза, но можно потом уменьшить
        output, _ = upsampler.enhance(img, outscale=outscale)
        return output, outscale / 4.0   # реальный коэффициент
    else:
        raise ValueError(f"Unknown upscale model: {model_name}")
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Путь к пустому кадру')
    parser.add_argument('--camera_id', required=True, help='Идентификатор камеры')
    parser.add_argument('--clahe', type=float, default=0, help='CLAHE clip limit (0 = выкл)')
    parser.add_argument('--denoise', type=float, default=0, help='Сила шумоподавления (0 = выкл)')
    parser.add_argument('--sharpen', type=float, default=0, help='Сила повышения резкости (0 = выкл)')
    #parser.add_argument('--upscale_model', default=None, help='Модель для увеличения разрешения (realesrgan)')
    #parser.add_argument('--upscale_factor', type=float, default=2.0, help='Во сколько раз увеличить')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f'Не найден файл {args.input}')
    orig_h, orig_w = img.shape[:2]

    # --- Предобработка (CLAHE, denoise, sharpen) ---
    if args.clahe > 0:
        img = apply_clahe(img, clip=args.clahe)
    if args.denoise > 0:
        img = apply_denoise(img, h=args.denoise)
    if args.sharpen > 0:
        img = apply_sharpen(img, amount=args.sharpen)

    # --- Опциональный апскейл нейросетью ---
    scale_factor = 1.0
    cv2.imwrite(f'enhanced_{args.camera_id}.png', img)
    # --- Загрузка MoGe и инференс ---
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.eval()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=device).permute(2,0,1)
    with torch.no_grad():
        output = model.infer(tensor)

    points = output['points'].cpu().numpy()   # (H, W, 3)
    mask   = output['mask'].cpu().numpy()     # (H, W)
    intrinsics = output['intrinsics'].cpu().numpy()  # (3,3)

    # --- Сохранение в папку камеры ---
    cfg_dir = Path('cfg') / args.camera_id
    cfg_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg_dir / 'points_map.npy', points)
    np.save(cfg_dir / 'mask.npy', mask)
    np.save(cfg_dir / 'intrinsics.npy', intrinsics)

    # Сохраняем исходное разрешение и масштаб, чтобы main.py мог пересчитать координаты bbox
    meta = {'original_width': orig_w, 'original_height': orig_h, 'scale_factor': scale_factor}
    import json
    with open(cfg_dir / 'meta.json', 'w') as f:
        json.dump(meta, f)

    print(f"Сцена для камеры {args.camera_id} сохранена в {cfg_dir}")
    print(f"Оригинальный размер: {orig_w}x{orig_h}, масштаб: {scale_factor}")
