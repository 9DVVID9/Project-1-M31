import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

url = "https://wxyzwebcams.com/network/horizon.php?id=4032"  # no #google_vignette
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": url,
}

def list_imgs(page_url):
    r = requests.get(page_url, headers=headers, timeout=12)
    print("\nPAGE:", page_url, "| status:", r.status_code)
    soup = BeautifulSoup(r.text, "html.parser")

    imgs = soup.find_all("img")
    print("  imgs found:", len(imgs))
    for i, im in enumerate(imgs[:10]):  # show up to 10
        src = im.get("src") or im.get("data-src")
        print(f"   [{i}] img src:", src, "->", urljoin(page_url, src) if src else None)

    ifrs = soup.find_all("iframe")
    print("  iframes found:", len(ifrs))
    for i, fr in enumerate(ifrs[:10]):  # check up to 10 iframes
        fr_src = fr.get("src")
        abs_fr = urljoin(page_url, fr_src) if fr_src else None
        print(f"   [{i}] iframe src:", fr_src, "->", abs_fr)
        if abs_fr:
            # fetch inside iframe and list images there
            try:
                r2 = requests.get(abs_fr, headers=headers, timeout=12)
                print("     [iframe] status:", r2.status_code)
                s2 = BeautifulSoup(r2.text, "html.parser")
                imgs2 = s2.find_all("img")
                print("     [iframe] imgs:", len(imgs2))
                for j, im2 in enumerate(imgs2[:10]):
                    s2src = im2.get("src") or im2.get("data-src")
                    print(f"       [{j}] iframe img src:", s2src, "->", urljoin(abs_fr, s2src) if s2src else None)
            except Exception as e:
                print("     [iframe] fetch error:", e)

list_imgs(url)
