from pydantic import BaseModel


# @dataclass(kw_only=True)
class DatasetDetails(BaseModel):
    data_packing: str = "pack"  # 'prepacked' | 'pad' | 'varlen' | 'pack' | 'sortpack' | 'buffersmartpack' | 'bufferpack'
    dataset: str = ""  # = 'data/fineweb10B/fineweb_val_*.bin'
    dataset_name: str | None = None
    split: str = "train"
    text_column: str = "text"
    shuffle_seed: int | None = 0
    min_document_chars: int | None = (
        None  # if not None, documents with fewer characters than this are filtered out from the dataset
    )
    min_document_tokens: int | None = (
        None  # if not None, documents with fewer tokens than this are filtered out from the dataset
    )
    max_document_tokens: int = 9999999  # docs are cropped to this length, in tokens
    sequence_length: int = 1024  # sequence length, in tokens
    device_batch_size: int = 64  # batch size, in sequences, per device
    range_begin: int | None = None
    range_end: int | None = None
