import antlr4

try:
    from dataset.codenn.csharp.py2x.CSharp4Lexer import CSharp4Lexer
except ImportError:
    from .CSharp4Lexer import CSharp4Lexer

import re


def parse_csharp_code(code):
    code = code.replace('\\n', '\n')
    parsedVersion = []
    stream = antlr4.InputStream(code)
    lexer = CSharp4Lexer(stream)
    toks = antlr4.CommonTokenStream(lexer)
    toks.fetch(500)

    for token in toks.tokens:
        if token.type == 9 or token.type == 7 or token.type == 6:  # whitespace and comments and newline
            pass
        else:
            parsedVersion += [str(token.text)]

    last_token = parsedVersion.pop(-1)
    if last_token != '<EOF>':
        parsedVersion.append(last_token)
    parsedVersion = [re.sub('\s+', ' ', token.strip()) for token in parsedVersion]
    return parsedVersion


def parse_csharp_docstring(docstring):
    docstring = docstring.strip().decode('utf-8').encode('ascii', 'replace')
    return re.findall(r"[\w]+|[^\s\w]", docstring)


if __name__ == '__main__':
    print(
        parse_csharp_code("public Boolean SomeValue {     get { return someValue; }     set { someValue = value; } }")
    )
    print(parse_csharp_code(
        "Console.WriteLine('cat'); int mouse = 5; int cat = 0.4; int cow = 'c'; int moo = \"mouse\"; ")
    )
    print(parse_csharp_code(
        "int i = 4;  // i is assigned the literal value of '4' \n int j = i   // j is assigned the value of i.  Since i is a variable,               //it can change and is not a 'literal'"))
    try:
        print(parse_csharp_code('string `fixed = Regex.Replace(input, "\s*()","$1");'))
    except:
        print("Error")
