from utils import generate_synthetic_train_data, load_synthetic_data

if __name__ == "__main__":
    generate_synthetic_train_data()
    data = load_synthetic_data()
    data = data[
        (data["status"] == "success") & (data["warnings"].apply(len) == 0)
    ]

    for row in data.to_dict("records"):
        print(row["text"], "\n")
        for span in row["spans"]:
            print(span["text"], span["start"], span["end"])
        print("-" * 60)

    print(data)
