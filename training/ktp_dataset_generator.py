"""
Synthetic KTP Dataset Generator
Generates fake KTP images + YOLO format labels + augmentation
Usage: python ktp_dataset_generator.py --count 500 --output ./dataset
"""

import os
import random
import shutil
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from faker import Faker
import albumentations as A
import numpy as np

from ktp_schema import BLOODS, JOBS, PROVINCES, RELIGIONS
from ktp_schema import MARITAL_STATUSES as MARITAL

# ─── Config ───────────────────────────────────────────────────────────────────

fake = Faker("id_ID")


def _find_font(candidates: list[str], fallback_name: str) -> str:
    """Returns the first existing font path, or a matplotlib-bundled fallback.

    Font locations differ across Linux containers, Windows dev machines, and
    CI images, so we probe common paths rather than hardcoding one.
    """
    for path in candidates:
        if os.path.exists(path):
            return path
    import matplotlib.font_manager as fm
    return fm.findfont(fallback_name, fontext="ttf")


FONT_REG = _find_font(
    [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ],
    "DejaVu Sans",
)
FONT_BOLD = _find_font(
    [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ],
    "DejaVu Sans:bold",
)

KTP_W, KTP_H = 640, 404   # card canvas size
IMG_W, IMG_H = 960, 720   # full scene size (card placed on background)
# Fixed column where every field's value starts, regardless of its label's
# length — a real KTP's label and value sit side-by-side on one row rather
# than stacked, and confirmed real labels ("Status Perkawinan",
# "Kewarganegaraan") fit comfortably before this column at the field font size.
VALUE_COLUMN_X = 250

STREETS = [
    "JL. MERDEKA", "JL. SUDIRMAN", "JL. GATOT SUBROTO",
    "JL. DIPONEGORO", "GG. MAWAR", "KOMP. GRIYA INDAH",
    "JL. AHMAD YANI", "JL. PAHLAWAN", "JL. IMAM BONJOL",
]
BG_COLORS = [
    (240, 235, 220),   # cream (standard)
    (235, 240, 235),   # slight green tint
    (245, 235, 225),   # warm cream
    (230, 230, 235),   # cool white
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def fake_nik() -> str:
    prov = random.randint(11, 99)
    city = random.randint(1,  99)
    dist = random.randint(1,  99)
    day  = random.randint(1,  28)
    mon  = random.randint(1,  12)
    yr   = random.randint(70, 99)
    seq  = random.randint(1,  9999)
    return f"{prov:02d}{city:02d}{dist:02d}{day:02d}{mon:02d}{yr:02d}{seq:04d}"

def fake_address() -> str:
    street = random.choice(STREETS)
    no     = random.randint(1, 150)
    rt     = random.randint(1, 20)
    rw     = random.randint(1, 10)
    return f"{street} NO.{no} RT {rt:03d}/RW {rw:03d}"

def truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars - 1] + "."


# ─── KTP Renderer ─────────────────────────────────────────────────────────────

def render_ktp() -> tuple[Image.Image, dict[str, str]]:
    """Renders a single fake KTP card as a PIL Image.

    Returns (image, ground_truth) where ground_truth maps ktp_schema
    KtpFields field names to the exact text printed on the card (post
    truncation, matching what OCR could actually read) — used by
    evaluate_pipeline.py to measure real extraction accuracy instead of
    eyeballing rendered images one at a time.
    """
    province = random.choice(PROVINCES)
    city     = fake.city().upper()
    nik      = fake_nik()
    name     = truncate(fake.name().upper(), 30)
    pob      = fake.city().upper()
    dob      = f"{random.randint(1,28):02d}-{random.randint(1,12):02d}-{random.randint(1970,2002)}"
    gender   = random.choice(["LAKI-LAKI", "PEREMPUAN"])
    address  = truncate(fake_address(), 38)
    rt_rw    = f"{random.randint(1,15):03d}/{random.randint(1,10):03d}"
    kel      = truncate(fake.city().upper(), 18)
    kec      = fake.city().upper()
    religion = random.choice(RELIGIONS)
    marital  = random.choice(MARITAL)
    job      = random.choice(JOBS)
    blood    = random.choice(BLOODS)
    bg_color = random.choice(BG_COLORS)

    img  = Image.new("RGB", (KTP_W, KTP_H), color=bg_color)
    draw = ImageDraw.Draw(img)

    # ── Header ────────────────────────────────────────────────────────────────
    draw.rectangle([(0, 0), (KTP_W, 60)], fill=(30, 60, 140))

    # Garuda stand-in
    draw.rectangle([(8, 4), (54, 56)], fill=(200, 170, 30))
    draw.text((31, 30), "G",
              font=ImageFont.truetype(FONT_BOLD, 22),
              fill=(30, 60, 140), anchor="mm")

    fh1 = ImageFont.truetype(FONT_BOLD, 13)
    fh2 = ImageFont.truetype(FONT_REG,  10)
    draw.text((KTP_W // 2, 15), "REPUBLIK INDONESIA",
              font=fh1, fill="white", anchor="mm")
    draw.text((KTP_W // 2, 31), province,
              font=fh1, fill="white", anchor="mm")
    draw.text((KTP_W // 2, 48), f"KOTA/KABUPATEN {truncate(city, 24)}",
              font=fh2, fill=(200, 220, 255), anchor="mm")

    # ── Photo box ─────────────────────────────────────────────────────────────
    photo_color = random.choice([(180,180,180),(160,170,185),(175,165,160)])
    draw.rectangle([(10, 70), (135, 220)], fill=photo_color, outline=(120,120,120))
    draw.text((72, 145), "FOTO",
              font=ImageFont.truetype(FONT_REG, 9),
              fill=(80, 80, 80), anchor="mm")
    draw.text((72, 232), "KTP",
              font=ImageFont.truetype(FONT_BOLD, 11),
              fill=(30, 60, 140), anchor="mm")

    # ── NIK ───────────────────────────────────────────────────────────────────
    # Label and value on the same row (label column, then a fixed value
    # column further right) rather than stacked on two lines — confirmed
    # against a real KTP photo to be the actual layout; the card's own
    # label/value boxes sit side-by-side on one visual row, not one above
    # the other. NIK keeps its larger/bolder value font (also matches the
    # real card's visual emphasis on this field).
    draw.text((145, 78), "NIK",
              font=ImageFont.truetype(FONT_BOLD, 11), fill=(30, 30, 100))
    draw.text((VALUE_COLUMN_X, 78), f": {nik}",
              font=ImageFont.truetype(FONT_BOLD, 14), fill=(10, 10, 10))

    # ── Fields ────────────────────────────────────────────────────────────────
    # RT/RW and Kel/Desa are separate rows (also confirmed against a real
    # card) — a prior version of this generator combined them onto one row,
    # which the OCR pipeline had never seen split and so silently failed to
    # extract kelurahan_desa on a real card.
    fields = [
        ("Nama",              name),
        ("Tempat/Tgl Lahir",  f"{truncate(pob,16)}, {dob}"),
        ("Jenis Kelamin",     f"{gender}   Gol. Darah: {blood}"),
        ("Alamat",            address),
        ("RT/RW",             rt_rw),
        ("Kel/Desa",          kel),
        ("Kecamatan",         kec),
        ("Agama",             religion),
        ("Status Perkawinan", marital),
        ("Pekerjaan",         job),
        ("Kewarganegaraan",   "WNI"),
        ("Berlaku Hingga",    "SEUMUR HIDUP"),
    ]

    fl = ImageFont.truetype(FONT_REG,  9)
    fv = ImageFont.truetype(FONT_BOLD, 9)
    y  = 100
    for label, value in fields:
        draw.text((145, y),             label,        font=fl, fill=(80, 80, 80))
        draw.text((VALUE_COLUMN_X, y),  f": {value}", font=fv, fill=(10, 10, 10))
        y += 17

    # ── Footer ────────────────────────────────────────────────────────────────
    draw.rectangle([(0, KTP_H - 6), (KTP_W, KTP_H)], fill=(30, 60, 140))

    ground_truth = {
        "nik": nik,
        "nama": name,
        "tempat_lahir": truncate(pob, 16),
        "tanggal_lahir": dob,
        "jenis_kelamin": gender,
        "golongan_darah": blood,
        "alamat": address,
        "rt_rw": rt_rw,
        "kelurahan_desa": kel,
        "kecamatan": kec,
        "agama": religion,
        "status_perkawinan": marital,
        "pekerjaan": job,
        "kewarganegaraan": "WNI",
        "berlaku_hingga": "SEUMUR HIDUP",
        "provinsi": province,
        "kota_kabupaten": truncate(city, 24),
    }
    return img, ground_truth


# ─── Hard-negative distractors ─────────────────────────────────────────────
#
# False-positive testing against real (non-KTP) photos found the detector,
# trained only on positive KTP examples, had learned "bordered rectangle
# containing lines of text" rather than anything KTP-specific — a plain
# colored rectangle was correctly ignored, but a generic business-card mockup
# (white rect, black border, unrelated text lines) triggered a false
# detection at *higher* confidence than most true positives. These
# distractors are deliberately card-shaped and text-bearing, but visually
# and structurally distinct from a KTP (different proportions, palette,
# layout, no "REPUBLIK INDONESIA"/Garuda emblem), so the model has to learn
# actual KTP-specific features instead of generic "rectangle with text."

DISTRACTOR_HEADER_COLORS = [
    (60, 130, 70),     # green (e.g. a membership/loyalty card)
    (150, 40, 40),      # red
    (90, 90, 95),       # slate gray
    None,                # no header bar at all
]
DISTRACTOR_TITLES = [
    "MEMBER CARD", "STUDENT ID", "EMPLOYEE PASS", "LIBRARY CARD",
    "ACME CORP", "LOYALTY CLUB", "ACCESS PASS",
]


def _random_words(n: int) -> str:
    return " ".join(fake.word() for _ in range(n)).upper()


def render_distractor() -> Image.Image:
    """Renders a random non-KTP, card-shaped or panel-shaped object — a
    hard negative for detection training. Returns an RGBA image so it can
    be rotated/pasted onto a scene the same way render_ktp()'s output is.
    """
    kind = random.choice(["card", "photo_panel", "receipt"])

    if kind == "card":
        w, h = random.randint(380, 700), random.randint(220, 440)
        bg = random.choice([(255, 255, 255), (245, 245, 240), (250, 248, 235)])
        img = Image.new("RGBA", (w, h), (*bg, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (w - 1, h - 1)], outline=(0, 0, 0), width=3)

        header = random.choice(DISTRACTOR_HEADER_COLORS)
        body_top = 0
        if header is not None:
            bar_h = int(h * 0.15)
            draw.rectangle([(0, 0), (w, bar_h)], fill=header)
            draw.ellipse([(10, 5), (10 + bar_h - 10, bar_h - 5)], fill=(220, 220, 220))
            draw.text((w // 2, bar_h // 2), random.choice(DISTRACTOR_TITLES),
                       font=ImageFont.truetype(FONT_BOLD, 14), fill="white", anchor="mm")
            body_top = bar_h + 15

        fl = ImageFont.truetype(FONT_REG, 10)
        y = body_top + 15
        for _ in range(random.randint(3, 6)):
            draw.text((20, y), _random_words(random.randint(2, 4)), font=fl, fill=(20, 20, 20))
            y += 22
            if y > h - 20:
                break
        return img

    if kind == "photo_panel":
        w, h = random.randint(200, 500), random.randint(200, 500)
        fill = (random.randint(60, 220), random.randint(60, 220), random.randint(60, 220))
        img = Image.new("RGBA", (w, h), (*fill, 255))
        draw = ImageDraw.Draw(img)
        if random.random() < 0.6:
            draw.rectangle([(0, 0), (w - 1, h - 1)], outline=(255, 255, 255), width=6)
        return img

    # receipt: narrow, many short lines
    w, h = random.randint(180, 260), random.randint(400, 650)
    img = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    fl = ImageFont.truetype(FONT_REG, 9)
    y = 15
    for _ in range(random.randint(10, 18)):
        draw.text((10, y), _random_words(random.randint(1, 3)), font=fl, fill=(10, 10, 10))
        y += 18
        if y > h - 15:
            break
    return img


# ─── Scene Composer ───────────────────────────────────────────────────────────

def compose_scene(ktp_img: Image.Image) -> tuple[Image.Image, tuple]:
    """
    Places the KTP card onto a random background scene.
    Returns (scene_image, bbox_xyxy) where bbox is the card's position.
    """
    scene = Image.new("RGB", (IMG_W, IMG_H),
                      color=(random.randint(30,200),
                             random.randint(30,200),
                             random.randint(30,200)))

    # Random slight rotation of card (-15 to +15 deg). Rotating in RGBA and
    # pasting with the alpha channel as a mask means only the tilted card's
    # own pixels land on the scene — the corners of rotate()'s bounding
    # square stay transparent, so the real scene background shows through
    # them instead of an artificial black square. (A solid fillcolor there
    # previously created a second, axis-aligned high-contrast edge that
    # confused contour-based deskewing downstream — it was picking up that
    # square instead of the card's true tilted silhouette.)
    angle   = random.uniform(-15, 15)
    rotated = ktp_img.convert("RGBA").rotate(angle, expand=True)

    rw, rh = rotated.size
    # Random position: card stays mostly inside scene
    max_x = max(0, IMG_W - rw)
    max_y = max(0, IMG_H - rh)
    px    = random.randint(0, max(0, max_x))
    py    = random.randint(0, max(0, max_y))

    scene.paste(rotated, (px, py), mask=rotated)

    # Bounding box of the card in scene coordinates (xyxy)
    x1, y1 = px, py
    x2, y2 = min(px + rw, IMG_W), min(py + rh, IMG_H)
    return scene, (x1, y1, x2, y2)


def compose_negative_scene() -> Image.Image:
    """Builds a background-only scene with no KTP present — either a plain
    background, or a background with a non-KTP distractor object placed on
    it (see render_distractor). No bounding box is returned: the caller
    writes an empty YOLO label file for these, the standard way to teach a
    detector what a true negative looks like.
    """
    scene = Image.new("RGB", (IMG_W, IMG_H),
                      color=(random.randint(30, 200),
                             random.randint(30, 200),
                             random.randint(30, 200)))

    if random.random() < 0.25:
        return scene  # plain background, nothing placed on it

    distractor = render_distractor()
    angle = random.uniform(-15, 15)
    rotated = distractor.rotate(angle, expand=True)

    rw, rh = rotated.size
    max_x = max(0, IMG_W - rw)
    max_y = max(0, IMG_H - rh)
    px = random.randint(0, max(0, max_x))
    py = random.randint(0, max(0, max_y))
    scene.paste(rotated, (px, py), mask=rotated)
    return scene


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h) -> str:
    """Converts xyxy bbox to YOLO normalized format string."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# ─── Augmentation Pipeline ────────────────────────────────────────────────────

augment = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.GaussianBlur(blur_limit=(3, 5), p=0.4),
    A.GaussNoise(p=0.3),
    A.RandomShadow(p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, p=0.4),
    A.Perspective(scale=(0.02, 0.06), p=0.5),
    A.ImageCompression(quality_range=(60, 95), p=0.4),
], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"],
                             clip=True, min_visibility=0.3))

# Same transforms, without bbox_params — negatives have no object of
# interest for Albumentations to track.
augment_negative = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.GaussianBlur(blur_limit=(3, 5), p=0.4),
    A.GaussNoise(p=0.3),
    A.RandomShadow(p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, p=0.4),
    A.ImageCompression(quality_range=(60, 95), p=0.4),
])


# ─── Main Generator ───────────────────────────────────────────────────────────

def generate_dataset(count: int, output_dir: str, aug_factor: int = 2, neg_ratio: float = 0.15):
    """
    Generates `count` base scenes then augments each `aug_factor` times.
    Total positive images = count * (1 + aug_factor); an additional
    negative (no-KTP) set of roughly count * neg_ratio base scenes is
    generated the same way, each labeled with an empty YOLO label file, so
    the detector sees true background/distractor examples during training
    rather than only ever seeing images that contain a KTP.
    """
    base_dir  = Path(output_dir)
    all_imgs  = []   # collect everything before splitting

    tmp_dir = base_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {count} base KTP scenes...")
    for i in range(count):
        ktp, _ground_truth = render_ktp()  # ground truth unused here — this dataset is for detection training only
        scene, bbox = compose_scene(ktp)
        x1, y1, x2, y2 = bbox

        stem = f"ktp_{i:05d}"

        # Save base image
        img_path = tmp_dir / f"{stem}.jpg"
        scene.save(img_path, quality=90)

        yolo_line = xyxy_to_yolo(x1, y1, x2, y2, IMG_W, IMG_H)
        all_imgs.append((img_path, yolo_line))

        # Augmented variants
        scene_np = np.array(scene)
        for j in range(aug_factor):
            try:
                result = augment(
                    image=scene_np,
                    bboxes=[[x1, y1, x2, y2]],
                    labels=[0]
                )
                aug_arr  = result["image"]
                aug_bbox = result["bboxes"]
                if not aug_bbox:
                    continue
                ax1, ay1, ax2, ay2 = aug_bbox[0]
                aug_img  = Image.fromarray(aug_arr)
                aug_stem = f"ktp_{i:05d}_aug{j}"
                aug_path = tmp_dir / f"{aug_stem}.jpg"
                aug_img.save(aug_path, quality=85)
                aug_yolo = xyxy_to_yolo(ax1, ay1, ax2, ay2, IMG_W, IMG_H)
                all_imgs.append((aug_path, aug_yolo))
            except Exception:
                pass

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{count} base images done...")

    neg_count = int(count * neg_ratio)
    print(f"Generating {neg_count} negative (no-KTP) scenes...")
    for i in range(neg_count):
        scene = compose_negative_scene()
        stem = f"neg_{i:05d}"

        img_path = tmp_dir / f"{stem}.jpg"
        scene.save(img_path, quality=90)
        all_imgs.append((img_path, ""))  # empty label = background image

        scene_np = np.array(scene)
        for j in range(aug_factor):
            try:
                aug_arr = augment_negative(image=scene_np)["image"]
                aug_img = Image.fromarray(aug_arr)
                aug_stem = f"neg_{i:05d}_aug{j}"
                aug_path = tmp_dir / f"{aug_stem}.jpg"
                aug_img.save(aug_path, quality=85)
                all_imgs.append((aug_path, ""))
            except Exception:
                pass

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{neg_count} negative images done...")

    # ── Train / Val / Test split ─────────────────────────────────────────────
    random.shuffle(all_imgs)
    total  = len(all_imgs)
    n_train = int(total * 0.70)
    n_val   = int(total * 0.20)

    splits = {
        "train": all_imgs[:n_train],
        "val":   all_imgs[n_train:n_train + n_val],
        "test":  all_imgs[n_train + n_val:],
    }

    for split, items in splits.items():
        img_dir   = base_dir / split / "images"
        label_dir = base_dir / split / "labels"
        img_dir.mkdir(parents=True,   exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for img_path, yolo_line in items:
            dst_img   = img_dir   / img_path.name
            dst_label = label_dir / (img_path.stem + ".txt")
            shutil.copy(img_path, dst_img)
            dst_label.write_text(yolo_line)

    shutil.rmtree(tmp_dir)

    # ── Write ktp.yaml ───────────────────────────────────────────────────────
    yaml_path = base_dir / "ktp.yaml"
    yaml_path.write_text(f"""\
path: {base_dir.resolve()}
train: train/images
val:   val/images
test:  test/images

nc: 1
names: [ktp]
""")

    n_negative = sum(1 for _, yolo_line in all_imgs if not yolo_line)
    print(f"\nDone! Dataset summary:")
    print(f"  Total images : {total} ({n_negative} negative / no-KTP)")
    print(f"  Train        : {len(splits['train'])}")
    print(f"  Val          : {len(splits['val'])}")
    print(f"  Test         : {len(splits['test'])}")
    print(f"  Config       : {yaml_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic KTP dataset generator")
    parser.add_argument("--count",      type=int, default=200,
                        help="Number of base KTP scenes (default: 200)")
    parser.add_argument("--output",     type=str, default="./dataset",
                        help="Output directory (default: ./dataset)")
    parser.add_argument("--aug-factor", type=int, default=2,
                        help="Augmented copies per base image (default: 2)")
    parser.add_argument("--neg-ratio", type=float, default=0.15,
                        help="Negative (no-KTP) base scenes as a fraction of --count (default: 0.15)")
    args = parser.parse_args()

    generate_dataset(args.count, args.output, args.aug_factor, args.neg_ratio)
