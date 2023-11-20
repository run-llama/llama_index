import argparse

from llama_index.llama_pack.download import LLAMA_HUB_URL, download_llama_pack


def main():
    # Create a parser for downloading llama-packs
    parser = argparse.ArgumentParser(description="Download llama-packs")
    parser.add_argument(
        "llama-pack-class",
        type=str,
        dest="llama_pack_class",
        help=(
            "The name of the llama-pack class you want to download, "
            "such as `GmailOpenAIAgentPack`."
        ),
    )
    parser.add_argument(
        "download-dir",
        type=str,
        dest="download_dir",
        help="Custom dirpath to download the pack into.",
    )
    parser.add_argument(
        "--llama-hub-url",
        type=str,
        default=LLAMA_HUB_URL,
        help="URL to llama hub.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help=(
            "If true, the local cache will be skipped and the pack "
            "will be fetched directly from the remote repo."
        ),
    )

    args = parser.parse_args()
    download_llama_pack(
        llama_pack_class=args.llama_pack_class,
        download_dir=args.download_dir,
        llama_hub_url=args.llama_hub_url,
        refresh_cache=args.refresh_cache,
    )
    print(f"Successfully downloaded {args.llama_pack_class} to {args.download_dir}")
