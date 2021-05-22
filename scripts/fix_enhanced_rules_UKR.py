import re
import numpy as np
from copy import deepcopy


def get_lines(sent):
    meta_lines = re.search('^(#.*\n)*', sent).group()
    no_meta_sent = sent[len(meta_lines):]
    lines = no_meta_sent.split('\n')
    return lines, meta_lines


def get_deps_nums(deps):
    nums, labels = [], []
    for dep in deps:
        nums.append(int(dep.split(':')[0]))
        labels.append(dep.split(':')[1])
    return nums, labels


def choose_closest_dep(parts):
    deps = parts[8].split('|')
    nums, labels = get_deps_nums(deps)
    diffs = [abs(num-int(parts[6])) for num in nums]
    return deps[np.argmin(diffs)]


def change_to_basic(parts):
    return parts[6] + ':' + parts[7]


def correct_dep_to_self(parts, lines, new_lines, ind, num, labels, j):
    for i, line in enumerate(lines):
        if str(num) + ':' + 'X' in line.split('\t')[8]:
            other_parts = line.split('\t')
            other_parts[8] = parts[8]
            new_lines[i] = '\t'.join(other_parts)
        elif str(num) + ':' + 'X_' + labels[j] in line.split('\t')[8]:
            other_parts = line.split('\t')
            other_parts[8] = parts[8]
            new_lines[i] = '\t'.join(other_parts)
    parts[8] = change_to_basic(parts)
    new_lines[ind] = '\t'.join(parts)
    return new_lines


# Add info to OBL & NMOD part
def find_case_lex(all_lines, num_case_str):
    case_line = re.search(r'(\n|^).*{0}\t.*?\n'.format(num_case_str), all_lines)
    if case_line:
        case_lex = case_line.group().split('\t')[2]
        case_num = case_line.group().split('\t')[0]
        fixed_str = '\t' + str(case_num.strip()) + '\tfixed'
        fixed = re.findall(r'((\n|^).*{0}\t.*?)'.format(fixed_str), all_lines)
        if fixed:
            fixed_lexs = [f[0].split('\t')[2] for f in fixed]
            full_case_lex = case_lex + '_' + '_'.join(fixed_lexs)
            case_lex = full_case_lex
    else:
        case_lex = ''
    return case_lex


def choose_case_name(obl_num, all_lines, gram):
    nummod = re.search(r'.*\t{0}\t{1}\t.*'.format(obl_num, 'nummod:gov'), all_lines)
    if nummod:
        nummod = nummod.group()
        num_gram = nummod.split('\t')[5]
        case_name = re.search('Case=(.*?)(\t|\|)', num_gram)
    else:
        case_name = re.search('Case=(.*?)(\t|\|)', gram)
    return case_name


def add_case_to_line(c_line, all_lines):
    c_line = c_line.strip()
    parts = c_line.split('\t')
    deps_list = parts[8].split('|')
    case_dep_inds = []
    added_conj = False
    conj_line = ''
    orig_conj_line = ''
    for i, dep in enumerate(deps_list):
        if 'obl' in dep or 'nmod' in dep:
            case_dep_inds.append(i)

    for ind in case_dep_inds:
        enh_dep = ':'.join(deps_list[ind].split(':')[:2])
        base_enh_dep = enh_dep  # preserve it without additions
        obl_num = parts[0]
        num_case_str = '\t' + str(obl_num) + '\tcase'
        case_lex = find_case_lex(all_lines, num_case_str)
        if case_lex != '':
            enh_dep = enh_dep + ':' + case_lex.strip()

        gram = parts[5]
        case_name = choose_case_name(obl_num, all_lines, gram)
        if case_name:
            case_name = case_name.group(1).lower()
            enh_dep = enh_dep + ':' + case_name
        deps_list[ind] = enh_dep
        if re.search(r'((\n|^).*((conj\|{0})|({0}.*\|\d+:conj)).*?\t.*?\n)'.format(base_enh_dep),
                     all_lines[all_lines.find(c_line):]):
            conj_line = re.search(r'((\n|^).*((conj\|{0})|({0}.*\|\d+:conj)).*?\t.*?\n)'.format(base_enh_dep),
                                  all_lines[all_lines.find(c_line):]).group()
            orig_conj_line = conj_line
            conj_parts = conj_line.split('\t')
            conj_deps = conj_parts[8].split('|')
            for j, dep in enumerate(conj_deps):
                if base_enh_dep in dep:
                    conj_deps[j] = enh_dep
            conj_parts[8] = '|'.join(conj_deps)
            conj_line = '\t'.join(conj_parts)
            added_conj = True
    parts[8] = '|'.join(deps_list)
    new_line = '\t'.join(parts)
    if not added_conj:
        conj_line = ''
    return new_line, conj_line, orig_conj_line


# for one sent
def add_case_to_sent(lines):
    for i, line in enumerate(lines):
        lines_text = '\n'.join(lines)
        parts = line.split('\t')
        if ('obl' in parts[8] or 'nmod' in parts[8]) and 'conj' not in parts[8]:
            lines[i], conj_line, orig_conj_line = add_case_to_line(line, lines_text)
            if conj_line:
                lines[lines.index(orig_conj_line.strip())] = conj_line.strip()
    return lines


# Add info to ADVCL & ACL part
def add_mark_to_line(c_line, all_lines):
    c_line = c_line.strip()
    added_conj = False
    orig_conj_line = ''
    conj_line = ''
    parts = c_line.split('\t')
    for dep in parts[8].split('|'):
        if ('acl' in dep and 'acl:relcl' not in dep) or 'advcl' in dep:
            enh_dep = ':'.join(dep.split(':')[:2])
            base_enh_dep = enh_dep
            break
    cl_num = parts[0]
    num_case_str = '\t' + str(cl_num) + '\tmark'
    case_lex = find_case_lex(all_lines, num_case_str)
    if case_lex != '':
        enh_dep = enh_dep + ':' + case_lex.strip()
    if re.search(r'((\n|^).*((conj\|{0})|({0}.*\|\d+:conj)).*?\t.*?\n)'.format(base_enh_dep),
                 all_lines[all_lines.find(c_line):]):
        conj_line = re.search(r'((\n|^).*((conj\|{0})|({0}.*\|\d+:conj)).*?\t.*?\n)'.format(base_enh_dep),
                              all_lines[all_lines.find(c_line):]).group()
        orig_conj_line = conj_line
        conj_parts = conj_line.split('\t')
        conj_deps = conj_parts[8].split('|')
        for j, dep in enumerate(conj_deps):
            if base_enh_dep in dep:
                conj_deps[j] = enh_dep
        conj_parts[8] = '|'.join(conj_deps)
        conj_line = '\t'.join(conj_parts)
        added_conj = True
    parts[8] = enh_dep
    new_line = '\t'.join(parts)
    if not added_conj:
        conj_line = ''
    return new_line, conj_line, orig_conj_line


def add_mark_to_sent(lines):
    for i, line in enumerate(lines):
        lines_text = '\n'.join(lines)  # this was made on purpose!
        parts = line.split('\t')
        if ('advcl' in parts[8] or ('acl' in parts[8] and 'acl:relcl' not in parts[8])) and 'conj' not in parts[8]:
            lines[i], conj_line, orig_conj_line = add_mark_to_line(line, lines_text)
            if conj_line:
                lines[lines.index(orig_conj_line.strip())] = conj_line.strip()
        parts = lines[i].split('\t')
        enh_deps = parts[8].split('|')
        try:
            parts[8] = '|'.join(sorted(enh_deps, key=lambda x: int(x.split(':')[0])))
        except:
            print(lines)
            exit()
        lines[i] = '\t'.join(parts)
    return lines


def delete_extra_nsubj(parts):
    simple_nsubj = []
    deps = parts[8].split('|')
    for dep in deps:
        if 'sp' not in dep and 'x' not in dep and 'rel' not in dep and 'nsubj' in dep:
            simple_nsubj.append(dep)
    if len(simple_nsubj) > 1:
        pseudo_parts = deepcopy(parts)
        pseudo_parts[8] = '|'.join(simple_nsubj)
        closest = choose_closest_dep(pseudo_parts)
        simple_nsubj.remove(closest)
        keep_deps = []
        for dep in deps:
            if dep not in simple_nsubj:
                keep_deps.append(dep)
        if not keep_deps:
            keep_deps.append(simple_nsubj[0])
        parts[8] = '|'.join(keep_deps)
    else:
        parts[8] = '|'.join(deps)
    return parts[8]


def first_part_fix(sent):
    lines, meta = get_lines(sent)
    new_lines = deepcopy(lines)
    sent_has_root = False
    for i, line in enumerate(lines):
        parts = line.split('\t')
        # double punct/case
        if parts[8].count('punct') > 1 or parts[8].count('case') > 1:
            parts[8] = choose_closest_dep(parts)
            new_lines[i] = '\t'.join(parts)

        # dep is head to itself
        dep_nums, labels = get_deps_nums(parts[8].split('|'))
        for j, num in enumerate(dep_nums):
            if str(num) == parts[0]:
                new_lines = correct_dep_to_self(parts, lines, new_lines, i, num, labels, j)
                break

        # punct doesn't have punct dep
        if parts[7] == 'punct' and 'punct' not in parts[8]:
            parts[8] = parts[6] + ':' + parts[7]
            new_lines[i] = '\t'.join(parts)

        # copy basic to deps with X
        parts = new_lines[i].split('\t')
        if 'X' in parts[8]:
            parts[8] = change_to_basic(parts)
            new_lines[i] = '\t'.join(parts)

        # delete ellipsis
        parts = new_lines[i].split('\t')
        if '>' in parts[8]:
            parts[8] = change_to_basic(parts)
            new_lines[i] = '\t'.join(parts)

        # nsubj came from conj
        parts = new_lines[i].split('\t')
        if parts[8].count('nsubj') > 1:
            parts[8] = delete_extra_nsubj(parts)
            new_lines[i] = '\t'.join(parts)

        if 'root' in parts[8]:
            sent_has_root = True
    return meta, new_lines, sent_has_root


def second_part_fix(new_lines, sent_has_root):
    uk_extra_deps = ['nsubj:sp', 'csubj:sp', 'xcomp:sp', 'nsubj:x', 'csubj:x', 'xcomp:x', 'nsubj:rel', 'csubj:rel',
                     'xcomp:rel', 'obj:rel', 'obl:rel', 'flat:foreign', 'flat:title', 'flat:range',
                     'parataxis:discourse']
    result_lines = deepcopy(new_lines)
    for i, line in enumerate(new_lines):
        parts = line.split('\t')

        # sent doesn't have root
        if parts[7] == 'root' and not sent_has_root:
            parts[8] = parts[6] + ':' + parts[7]
            result_lines[i] = '\t'.join(parts)

        # replace special ukranian deps
        parts = result_lines[i].split('\t')
        deps = parts[8].split('|')
        new_deps = []
        for dep in deps:
            no_num_dep = ':'.join(dep.split(':')[1:])
            if no_num_dep in uk_extra_deps:
                if no_num_dep == 'flat:title':
                    dep = dep.split(':')[0] + ':' + 'appos'
                else:
                    dep = ':'.join(dep.split(':')[:-1])
            new_deps.append(dep)
        parts[8] = '|'.join(new_deps)
        parts[8] = parts[8].replace('det:numgov', 'advmod').replace('advmod:det', 'det')
        result_lines[i] = '\t'.join(parts)

    result_lines = add_mark_to_sent(result_lines)
    result_lines = add_case_to_sent(result_lines)
    return '\n'.join(result_lines)


def delete_deps_with_ref(new_sents):  # for easier evaluation
    for i, sent in enumerate(new_sents):
        lines, meta = get_lines(sent)
        for j, line in enumerate(lines):
            parts = line.split('\t')
            if 'ref' in parts[8]:
                deps = parts[8].split('|')
                for dep in deps:
                    if 'ref' in dep:
                        parts[8] = dep
                        break
            lines[j] = '\t'.join(parts)
        new_sents[i] = '# sent_id = ' + str(i) + '\n' + meta + '\n'.join(lines)
    return new_sents


def delete_extra_root(sent):
    lines, meta = get_lines(sent)
    root_found = False
    root_line = ''
    for j, line in enumerate(lines):
        parts = line.split('\t')
        if 'root' in parts[8] and 'root' in parts[7]:  # check if some root is in place
            root_found = True
            root_line = line
    root_added = False
    for j, line in enumerate(lines):
        parts = line.split('\t')
        if not root_found and 'root' in parts[8] and not root_added:  # if one root is in right place
            root_added = True
            continue
        elif not root_found and 'root' in parts[8] and root_added:
            parts[8] = change_to_basic(parts)
        elif root_found and 'root' in parts[8] and line == root_line:  # no roots in place
            continue
        elif root_found and 'root' in parts[8] and line != root_line:
            parts[8] = change_to_basic(parts)
        lines[j] = '\t'.join(parts)
    return meta + '\n'.join(lines)


def fix_rules(token_align_file, output_file):
    with open(token_align_file, 'r', encoding='utf-8') as f:
        text = f.read()
    sents = text.split('\n\n')[:-1]
    new_sents = []
    for i, sent in enumerate(sents):
        meta, new_lines, sent_has_root = first_part_fix(sent)
        new_sent = second_part_fix(new_lines, sent_has_root)
        if new_sent.count('root') > 2:
            new_sent = delete_extra_root(new_sent)
        if new_sent.count('root') > 2:
            print(new_sent)
        new_sents.append(meta + new_sent)
    new_sents = delete_deps_with_ref(new_sents)
    with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n\n'.join(new_sents) + '\n\n')


# fix_rules()
