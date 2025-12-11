import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import os


def fetch_image_from_url(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    arr = np.asarray(bytearray(r.content), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Brak obrazu z URL")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def show_histograms(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])

    colors = ("r", "g", "b")
    hists_rgb = [cv2.calcHist([image_rgb], [i], None, [256], [0, 256]) for i in range(3)]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor="lightgreen")
    for row in axes:
        for ax in row:
            ax.set_facecolor("lightgreen")

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Oryginal")
    axes[0, 0].axis("off")

    axes[0, 1].plot(hist_gray, color="black")
    axes[0, 1].set_title("Histogram jasnosci")
    axes[0, 1].set_xlim([0, 256])

    for hist, color in zip(hists_rgb, colors):
        axes[1, 0].plot(hist, color=color, label=color.upper())
    axes[1, 0].set_title("RGB")
    axes[1, 0].set_xlim([0, 256])
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    axes[1, 1].text(0.1, 0.5, "Wynik w konsoli.", fontsize=11)

    plt.tight_layout()
    plt.show()


def analyze_quality(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mean_val, std_val = cv2.meanStdDev(gray)
    mean_val = mean_val[0][0]
    std_val = std_val[0][0]

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    black_pixels = hist[0][0]
    white_pixels = hist[255][0]
    total_pixels = gray.shape[0] * gray.shape[1]
    black_pct = (black_pixels / total_pixels) * 100
    white_pct = (white_pixels / total_pixels) * 100

    d = []
    fix = False

    if mean_val < 80:
        d.append("Zdjecie nedoswietlone.")
        fix = True
    elif mean_val > 180:
        d.append("Zdjecie przeswietlone.")
        fix = True
    else:
        d.append("Jasnosc ok.")

    if std_val < 40:
        d.append("Niski kntrast.")
        fix = True
    else:
        d.append("Kontrast ok.")

    if black_pct > 2:
        d.append("Duzo czerni.")
    if white_pct > 2:
        d.append("Duzo bieli.")

    return {
        "mean": mean_val,
        "std": std_val,
        "black": black_pct,
        "white": white_pct,
        "diag": d,
        "fix": fix,
    }


def improve_with_clahe(image_rgb):
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def main():
    url = "https://upload.wikimedia.org/wikipedia/commons/b/b0/N%C3%B8rre_Vorup%C3%B8r_Coast_one_third_sky_2012-11-18.jpg"
    print("Pobieram...")
    try:
        image = fetch_image_from_url(url)
        print("Zdalne OK.")
    except Exception as e:
        print("URL blad:", e)
        img_bgr = cv2.imread("Coast.jpg", cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError("Brak pliku Coast.jpg")
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        print("Lokalne OK.")

    show_histograms(image)

    stats = analyze_quality(image)
    print("\nAnaliza:")
    print("Jasnosc:", round(stats["mean"], 2))
    print("Kontrast:", round(stats["std"], 2))
    print("Czern %:", round(stats["black"], 2))
    print("Biel % :", round(stats["white"], 2))
    for line in stats["diag"]:
        print("-", line)

    if stats["fix"]:
        print("\nKorekcja CLAHE.")
        improved = improve_with_clahe(image)
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), facecolor="lightgreen")
        for ax in axes:
            ax.set_facecolor("lightgreen")
        axes[0].imshow(image)
        axes[0].set_title("Oryginal")
        axes[0].axis("off")
        axes[1].imshow(improved)
        axes[1].set_title("Po CLAHE")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()
    else:
        print("\nBez korekcji.")


if __name__ == "__main__":
    main()
