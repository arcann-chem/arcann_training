def replace_in_list(input_list: list, substring_in: str, substring_out: str) -> list:
    """_summary_

    Args:
        input_list (list): input list of strings
        substring_in (str): string to replace
        substring_out (str): new string

    Returns:
        list: output list of strings
    """
    output_list = [f.replace(substring_in, substring_out) for f in input_list]
    return output_list


def delete_in_list(input_list: list, substring_in: str) -> list:
    """_summary_

    Args:
        input_list (list): input list of strings
        substring_in (str): substring to look for and delete the whole string

    Returns:
        list: output list of strings
    """
    output_list = [zzz for zzz in input_list if substring_in not in zzz]
    return output_list
