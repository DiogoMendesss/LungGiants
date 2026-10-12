from data.datamodule import NoduleDataModule


def main():
    dm = NoduleDataModule(batch_size=2, num_workers=0)
    dm.setup("fit")

    print("Training cases:", len(dm.train_dataset))
    print("Validation cases:", len(dm.val_dataset))

    batch = next(iter(dm.train_dataloader()))

    print("Cases:", batch["case_id"])
    print("Image batch:", batch["image"].shape)
    print("Mask batch:", batch["label"].shape)
    print("Mask values:", batch["label"].unique())


if __name__ == "__main__":
    main()