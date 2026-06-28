from utils.elliptic_loader import load_elliptic, print_elliptic_summary


DATA_DIR = "data/elliptic"


def main():
    data = load_elliptic(DATA_DIR)
    print_elliptic_summary(data)


if __name__ == "__main__":
    main()