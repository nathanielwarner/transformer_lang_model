import re


def preprocess(x: str, remove_stars=False, remove_java_doc_vars=False, remove_html_tags=False, remove_comments=False,
               remove_start_and_end_quotes=False, lower=False, to_edinburgh_format=False, remove_whitespace=False) -> str:
    if to_edinburgh_format:
        if x.endswith('\n'):
            x = x[:-len('\n')]
        x = x.replace('\n', ' DCNL ')
        x = x.replace('    ', ' DCSP ')
        x = x.replace('\t', ' DCSP ')
    if remove_java_doc_vars:
        x = re.sub(r'(?<![{])(@[\s\S]*)', ' ', x)
    if remove_comments:
        x = re.sub(r'(?<![:\"])(//.*?(?:\n|\\n))', ' ', x)
    if remove_html_tags:
        x = re.sub(r'<.*?>', ' ', x)
    if remove_whitespace:
        x = x.replace('\\n', ' ').replace('\n', ' ')
        x = x.replace('\\t', ' ').replace('\t', ' ')
    if remove_stars:
        x = x.replace('/*', ' ').replace('*/', ' ').replace('*', ' ')
    if remove_start_and_end_quotes:
        x = x.strip()
        if x.startswith("'"):
            x = x[len("'"):]
        if x.endswith("'"):
            x = x[:-len("'")]
        if x.startswith('"'):
            x = x[len('"'):]
        if x.endswith('"'):
            x = x[:-len('"')]
    x = x.strip()
    x = re.sub(r'(\s\s+)', ' ', x)
    if lower:
        x = x.lower()
    return x


def preprocess_csharp_or_java(x: str) -> str:
    return preprocess(x, remove_comments=True, remove_start_and_end_quotes=True, remove_whitespace=True)


def preprocess_javadoc(x: str) -> str:
    return preprocess(x, remove_stars=True, remove_java_doc_vars=True, remove_html_tags=True,
                      remove_start_and_end_quotes=True)
