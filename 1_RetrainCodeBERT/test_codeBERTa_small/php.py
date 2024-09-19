import pprint
PHP_CODE = """
public static <mask> set(string $key, $value) {
    if (!in_array($key, self::$allowedKeys)) {
        throw new \InvalidArgumentException('Invalid key given');
    }
    self::$storedValues[$key] = $value;
}
""".lstrip()
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="huggingface/CodeBERTa-small-v1",
    tokenizer="huggingface/CodeBERTa-small-v1"
)

pprint.pprint(fill_mask(PHP_CODE))
print()
pprint.pprint(fill_mask("My name is <mask>."))
