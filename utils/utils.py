
def txt_file_2_string(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        text = ''
        for line in lines:
            if not line.startswith('#'):
                text += line
    return text


def add_line_breaks(input_string, line_length):
    words = input_string.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= line_length:
            current_line += " " + word
        else:
            lines.append(current_line.strip())
            current_line = word
    lines.append(current_line.strip())
    return "\n".join(lines)


def lb(text):
    # Split the long string into lines
    lines = text.split('\n')
    # Remove any leading or trailing whitespace from each line
    lines = [line.strip() for line in lines]
    str = ""
    for line in lines:
        str = str + add_line_breaks(line, 75) + "\n"
    return str.strip()


def print_in_color(text, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
    }
    end_color = '\033[0m'
    if color in colors:
        print(f"{colors[color]}{text}{end_color}")
    else:
        print(text)