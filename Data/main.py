import os
import requests
import multiprocessing as mul
import os
import tarfile
import shutil
from tqdm import tqdm

SOURCE_DIR = "./Origin/"

url_list = []

http_header = {
    "Dnt": "1", "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
}


def get(url: str):
    save_path = os.path.join("./Origin/", url.split('/')[-1])
    response = requests.get(url=url, headers=http_header, timeout=10)
    if (response.status_code == 200):
        fp = open(save_path, "wb")
        fp.write(response.content)
        fp.close()
    return response.status_code


if __name__ == "__main__":
    os.makedirs(SOURCE_DIR, exist_ok=True)
    for postfix in ["logvol.+12.txt", "logvol.-12.txt", "tok.tgz"]:
        for year in range(1996, 2007):
            url_list.append("https://www.cs.cmu.edu/~ark/10K/data/%d.%s" % (year, postfix))
    pool = mul.Pool(mul.cpu_count())
    res = []
    for i in range(len(url_list)):
        url = url_list[i]
        res.append(pool.apply_async(get, (url, )))
    pool.close()
    for i in range(len(res)):
        print("%2d/%d %s %d." % (i + 1, len(res), url_list[i].split('/')[-1], res[i].get()))
