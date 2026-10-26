from src.argus.inference.remote import RemoteInference


def main():
    inference = RemoteInference()

    result = inference.analyze_text(
        "This technology could make document analysis much faster."
    )

    print("Remote inference result:")
    print(result)


if __name__ == "__main__":
    main()