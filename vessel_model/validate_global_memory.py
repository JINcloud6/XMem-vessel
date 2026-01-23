import argparse
import sys

from vessel_model.main import run_segmentation, xmem_config


def parse_list(value, cast=str):
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [cast(v) for v in items]


def main():
    parser = argparse.ArgumentParser(description="Validate global memory injection strategies.")
    parser.add_argument("--methods", default="all,nearest,similarity",
                        help="Comma-separated methods: all, nearest, similarity")
    parser.add_argument("--ks", default="50,100,200",
                        help="Comma-separated top-k values")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print configurations without running segmentation")
    args, remaining = parser.parse_known_args()

    methods = parse_list(args.methods, cast=str)
    ks = parse_list(args.ks, cast=int)

    for method in methods:
        for k in ks:
            xmem_config["enable_global_memory"] = True
            xmem_config["global_mem_select_method"] = method
            xmem_config["global_mem_topk"] = k
            print(f"[GlobalMemory] method={method} topk={k}")
            if args.dry_run:
                continue
            sys.argv = [sys.argv[0]] + remaining
            run_segmentation()


if __name__ == "__main__":
    main()
