"""""""""Download and prepare ASVspoof2019 dataset into simple data/real and data/fake folders.

Script to prepare ASVspoof 2019 dataset for training.

Download dataset from: https://datashare.ed.ac.uk/handle/10283/3336Script to prepare ASVspoof 2019 dataset for training.

"""

Download dataset from: https://datashare.ed.ac.uk/handle/10283/3336Usage examples:

import os

import pandas as pd"""  # If you already downloaded the archive:

import shutil

  python scripts/prepare_asvspoof2019.py --archive path/to/ASVspoof2019_LA_dev.tar.gz --out data/asvspoof2019

def prepare_asvspoof2019():

    """Prepare ASVspoof 2019 dataset by organizing audio files"""import os

    

    # Source paths (modify these according to your download location)import pandas as pd  # Or provide a direct URL (will attempt to download):

    base_path = "ASVspoof2019"

    protocol_path = os.path.join(base_path, "LA/ASVspoof2019_LA_cm_protocols")import shutil  python scripts/prepare_asvspoof2019.py --archive https://example.com/ASVspoof2019_LA_dev.tar.gz --out data/asvspoof2019

    audio_path = os.path.join(base_path, "LA/ASVspoof2019_LA_train/flac")

    

    # Output paths

    output_base = "datasets/asvspoof2019"def prepare_asvspoof2019():The script will extract the archive (zip or tar), locate audio files and protocol files (if present),

    train_path = os.path.join(output_base, "train")

    test_path = os.path.join(output_base, "test")    """Prepare ASVspoof 2019 dataset by organizing audio files"""and create a minimal directory structure:

    

    # Create output directories      <out>/real/*.wav

    for path in [train_path, test_path]:

        os.makedirs(os.path.join(path, "real"), exist_ok=True)    # Source paths (modify these according to your download location)  <out>/fake/*.wav

        os.makedirs(os.path.join(path, "fake"), exist_ok=True)

        base_path = "ASVspoof2019"

    # Process training data

    train_protocol = pd.read_csv(    protocol_path = os.path.join(base_path, "LA/ASVspoof2019_LA_cm_protocols")If a protocol file is available it will be used to assign labels; otherwise the script will try

        os.path.join(protocol_path, "ASVspoof2019.LA.cm.train.trn.txt"), 

        sep=" ", header=None    audio_path = os.path.join(base_path, "LA/ASVspoof2019_LA_train/flac")to infer labels from folder names or filename patterns.

    )

        """

    # Process each file

    for _, row in train_protocol.iterrows():    # Output pathsfrom __future__ import annotations

        filename = row[1] + ".flac"

        is_spoof = row[4] == "spoof"    output_base = "datasets/asvspoof2019"import argparse

        

        # Source and destination paths    train_path = os.path.join(output_base, "train")import os

        src = os.path.join(audio_path, filename)

        dst = os.path.join(    test_path = os.path.join(output_base, "test")import shutil

            train_path,

            "fake" if is_spoof else "real",    import sys

            filename

        )    # Create output directoriesimport tempfile

        

        # Copy file    for path in [train_path, test_path]:import tarfile

        if os.path.exists(src):

            shutil.copy2(src, dst)        os.makedirs(os.path.join(path, "real"), exist_ok=True)import zipfile

    

    print("Dataset preparation completed!")        os.makedirs(os.path.join(path, "fake"), exist_ok=True)from pathlib import Path

    print(f"Output directory: {output_base}")

    from typing import Dict, Optional

if __name__ == "__main__":

    prepare_asvspoof2019()    # Process training data

    train_protocol = pd.read_csv(try:

        os.path.join(protocol_path, "ASVspoof2019.LA.cm.train.trn.txt"),     import requests

        sep=" ", header=Noneexcept Exception:

    )    requests = None

    

    # Process each file

    for _, row in train_protocol.iterrows():def download_file(url: str, dest: Path) -> Path:

        filename = row[1] + ".flac"    if requests is None:

        is_spoof = row[4] == "spoof"        raise RuntimeError("requests is required to download files; please install it in your env")

            dest.parent.mkdir(parents=True, exist_ok=True)

        # Source and destination paths    print(f"Downloading {url} to {dest}...")

        src = os.path.join(audio_path, filename)    with requests.get(url, stream=True) as r:

        dst = os.path.join(        r.raise_for_status()

            train_path,        with open(dest, 'wb') as f:

            "fake" if is_spoof else "real",            for chunk in r.iter_content(chunk_size=8192):

            filename                if chunk:

        )                    f.write(chunk)

            print("Download finished")

        # Copy file    return dest

        if os.path.exists(src):

            shutil.copy2(src, dst)

    def extract_archive(archive: Path, extract_to: Path) -> Path:

    print("Dataset preparation completed!")    print(f"Extracting {archive} to {extract_to}...")

    print(f"Output directory: {output_base}")    extract_to.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive):

if __name__ == "__main__":        with zipfile.ZipFile(archive, 'r') as z:

    prepare_asvspoof2019()            z.extractall(path=extract_to)
    else:
        try:
            with tarfile.open(archive, 'r:*') as t:
                t.extractall(path=extract_to)
        except tarfile.ReadError:
            raise RuntimeError(f"Unsupported archive format: {archive}")
    print("Extraction complete")
    return extract_to


def find_protocol_file(root: Path) -> Optional[Path]:
    # Look for known protocol filenames or any .txt under protocols/
    candidates = list(root.rglob('*protocol*.txt')) + list(root.rglob('protocols/*.txt'))
    if candidates:
        return candidates[0]
    # fallback: any .txt that contains tokens 'genuine' or 'spoof'
    for p in root.rglob('*.txt'):
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(2048).lower()
                if 'genuine' in sample or 'spoof' in sample or 'bonafide' in sample:
                    return p
        except Exception:
            continue
    return None


def parse_protocol(protocol_path: Path) -> Dict[str, str]:
    """Parse a protocol file and return mapping utt_id -> label ('real' or 'fake').

    This parser is heuristic: it expects each line to contain an utterance id and a label
    token such as 'genuine'/'bonafide' or 'spoof'/'attack'.
    """
    mapping: Dict[str, str] = {}
    with open(protocol_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            toks = line.split()
            # find token that looks like wav id (contains '_' or endswith .wav)
            utt = toks[0]
            # some protocol files have utt without extension
            utt = utt if utt.lower().endswith('.wav') else utt + '.wav'
            label = None
            for t in toks[1:]:
                tl = t.lower()
                if 'genuine' in tl or 'bonafide' in tl or 'real' in tl:
                    label = 'real'
                    break
                if 'spoof' in tl or 'attack' in tl or 'fake' in tl:
                    label = 'fake'
                    break
            if label is None:
                # if token count >=2, try second token as label
                if len(toks) >= 2:
                    tl = toks[1].lower()
                    if tl in ('genuine', 'bonafide', 'real'):
                        label = 'real'
                    elif tl in ('spoof', 'attack', 'fake'):
                        label = 'fake'
            if label:
                mapping[utt] = label
    return mapping


def prepare_dataset(extracted_root: Path, out_dir: Path, protocol: Optional[Path] = None):
    print(f"Preparing dataset from {extracted_root} into {out_dir}")
    out_real = out_dir / 'real'
    out_fake = out_dir / 'fake'
    out_real.mkdir(parents=True, exist_ok=True)
    out_fake.mkdir(parents=True, exist_ok=True)

    mapping: Dict[str, str] = {}
    if protocol:
        print(f"Parsing protocol file {protocol}")
        mapping = parse_protocol(protocol)
        print(f"Protocol mapping contains {len(mapping)} entries")

    # gather audio files
    audio_exts = ('.wav', '.flac', '.mp3', '.m4a', '.wav')
    files = list(extracted_root.rglob('*'))
    audio_files = [p for p in files if p.suffix.lower() in audio_exts]
    print(f"Found {len(audio_files)} audio files under extracted root")

    moved = 0
    for p in audio_files:
        name = p.name
        label = None
        if name in mapping:
            label = mapping[name]
        else:
            # try without extension
            if name.endswith('.wav') and name[:-4] in mapping:
                label = mapping[name[:-4]]
        if label is None:
            # heuristic by folder name
            parts = [part.lower() for part in p.parts]
            if any(x in parts for x in ('fake', 'spoof')):
                label = 'fake'
            elif any(x in parts for x in ('real', 'genuine', 'bonafide')):
                label = 'real'
            else:
                # fallback: treat anything under 'ASVspoof' with 'LA' naming where filename contains 'spoof' unlikely
                if 'spoof' in name.lower() or 'fake' in name.lower():
                    label = 'fake'
                else:
                    label = 'real'

        dest = out_real if label == 'real' else out_fake
        try:
            shutil.copy2(p, dest / name)
            moved += 1
        except Exception as e:
            print(f"Failed to copy {p}: {e}")

    print(f"Copied {moved} files into {out_dir} (real: {len(list(out_real.iterdir()))}, fake: {len(list(out_fake.iterdir()))})")


def main():
    parser = argparse.ArgumentParser(description='Prepare ASVspoof2019 archive into data/real and data/fake')
    parser.add_argument('--archive', required=True, help='Path or URL to archive containing ASVspoof2019 files')
    parser.add_argument('--out', default='data/asvspoof2019', help='Output folder for prepared dataset')
    parser.add_argument('--protocol', default=None, help='Optional explicit protocol file to use')
    args = parser.parse_args()

    archive = Path(args.archive)
    out = Path(args.out)

    workdir = Path(tempfile.mkdtemp(prefix='asvsprep_'))
    try:
        if str(archive).lower().startswith('http'):
            if requests is None:
                print('requests is not installed in this environment. Please install requests or download the archive manually.')
                sys.exit(1)
            local_archive = workdir / Path(archive).name
            download_file(str(archive), local_archive)
            archive_path = local_archive
        else:
            archive_path = archive

        extracted = extract_archive(archive_path, workdir / 'extracted')

        protocol_path = None
        if args.protocol:
            protocol_path = Path(args.protocol)
        else:
            pf = find_protocol_file(Path(extracted))
            if pf:
                protocol_path = pf

        prepare_dataset(Path(extracted), out, protocol=protocol_path)
    finally:
        # keep workdir for inspection; do not auto-delete
        print(f"Temporary working directory retained at {workdir} for inspection")


if __name__ == '__main__':
    main()
