import re


def get_children(head_num, lines, depth=0):
    if depth == 2:
        return []
    children = []
    for line in lines:
        if line[0] == '#':
            continue
        parts = line.split('\t')
        if parts[6] == head_num:
            children.append((line, depth))
            children.extend(get_children(parts[0], lines, depth+1))
    return children


def filter_pos_children(children):
    candidates = []
    pos_candidates = ['ADV', 'DET', 'PRON']
    for child in children:
        parts = child[0].split('\t')
        if parts[3] in pos_candidates:
            candidates.append(child)
    return candidates


def check_rules(candidates):
    ref_lemmas = ['што', 'дзе', 'хто']
    if not candidates:
        return ''
    for c in candidates:
        parts = c[0].split('\t')
        if 'PronType=Rel' in parts[5]:
            return c[0]
        elif parts[2] in ref_lemmas:
            return c[0]
    return ''


def get_acl_head(lines, acl_head_num):
    for i, line in enumerate(lines):
        if line[0] == '#':
            continue
        if line.split('\t')[0] == acl_head_num:
            return line
    return None


def apply_acl_rule_to_sent(sent):
    lines = sent.split('\n')
    for i, line in enumerate(lines):
        if line[0] == '#':
            continue
        if 'acl' in line.split('\t')[7]:
            children = get_children(line.split('\t')[0], lines)
            candidates = filter_pos_children(children)
            ref_line = check_rules(candidates)
            if ref_line:
                ref_parts = ref_line.split('\t')
                acl_head_num = line.split('\t')[6]
                acl_head = get_acl_head(lines, acl_head_num)
                acl_head_parts = acl_head.split('\t')
                acl_head_parts[8] = acl_head_parts[8] + '|' + ref_parts[8]
                new_acl_head = '\t'.join(acl_head_parts)
                ref_parts[8] = acl_head.split('\t')[0] + ':ref'
                new_ref_line = '\t'.join(ref_parts)
                lines[lines.index(acl_head)] = new_acl_head
                lines[lines.index(ref_line)] = new_ref_line
    return '\n'.join(lines)


def same_animacy(acl_head, ref_conj, ref):
    head_gram = acl_head.split('\t')[5]
    conj_gram = ref_conj.split('\t')[5]
    ref_gram = []
    if ref != '':
        ref_gram = ref.split('\t')[5]
    animacy = 'Animacy=None'
    if 'Animacy' not in conj_gram:
        return True
    for gram in conj_gram.split('|'):
        if 'Animacy' in gram:
            animacy = gram
    if animacy in head_gram or animacy in ref_gram:
        return True
    else:
        return False


def apply_acl_rule_to_conj(sent):
    meta_lines = re.search('^(#.*\n)*', sent).group()
    no_meta_sent = sent[len(meta_lines):]
    lines = no_meta_sent.split('\n')
    for i, line in enumerate(lines):
        if line[0] == '#':
            continue
        if 'acl' in line.split('\t')[7]:
            acl_num = line.split('\t')[0]
            for l in lines:
                if '\t{}\tconj'.format(acl_num) in l:
                    acl_conj_line = l
                    children = get_children(acl_conj_line.split('\t')[0], lines)
                    candidates = filter_pos_children(children)
                    ref_line = check_rules(candidates)
                    if ref_line:
                        acl_head_num = line.split('\t')[6]
                        acl_head = get_acl_head(lines, acl_head_num)
                        children = get_children(line.split('\t')[0], lines)
                        candidates = filter_pos_children(children)
                        prev_ref_line = check_rules(candidates)
                        if same_animacy(acl_head, ref_line, prev_ref_line):
                            ref_parts = ref_line.split('\t')
                            acl_head_parts = acl_head.split('\t')
                            acl_head_parts[8] = acl_head_parts[8] + '|' + ref_parts[8]
                            new_acl_head = '\t'.join(acl_head_parts)
                            ref_parts[8] = acl_head.split('\t')[0] + ':ref'
                            new_ref_line = '\t'.join(ref_parts)
                            lines[lines.index(acl_head)] = new_acl_head
                            lines[lines.index(ref_line)] = new_ref_line
                            print('\n'.join(lines))
    return meta_lines + '\n'.join(lines)


def copy_basic_deps(line):
    parts = line.split('\t')
    parts[8] = parts[6] + ':' + parts[7]
    return '\t'.join(parts)


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
    enh_dep = parts[8]
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

    parts[8] = enh_dep
    new_line = '\t'.join(parts)
    return new_line


def add_case_to_all(all_lines, dep_type):
    dep_lines = re.findall(r'.*\t{0}\t.*'.format(dep_type), all_lines)
    new_lines = all_lines
    for dep_line in dep_lines:
        new_lines = new_lines.replace(dep_line, add_case_to_line(dep_line, all_lines))
    return new_lines



# MARK - ADVCL, ACL
def add_mark_to_line(c_line, all_lines):
    c_line = c_line.strip()
    parts = c_line.split('\t')
    enh_dep = parts[8]
    cl_num = parts[0]
    num_case_str = '\t' + str(cl_num) + '\tmark'
    case_lex = find_case_lex(all_lines, num_case_str)
    if case_lex != '':
        enh_dep = enh_dep + ':' + case_lex.strip()

    parts[8] = enh_dep
    new_line = '\t'.join(parts)
    return new_line


def add_mark_to_all(all_lines, dep_type):
    dep_lines = re.findall(r'.*\t{0}\t.*'.format(dep_type), all_lines)
    new_lines = all_lines
    for dep_line in dep_lines:
        new_lines = new_lines.replace(dep_line, add_mark_to_line(dep_line, all_lines))
    return new_lines


# CONJ
def add_conj_dep(dep_line, all_lines):
    parts = dep_line.split('\t')
    dep_enh = parts[8]

    dep_head = parts[6]
    head_line = re.search(r'(\n|^){0}\t.*'.format(dep_head), all_lines).group()
    head_enh = head_line.split('\t')[8]

    if '|' in head_enh:
        new_enhs = []
        enhs = head_enh.split('|')
        for e in enhs:
            if 'conj' not in e:
                new_enhs.append(e)
        head_enh = '|'.join(new_enhs)

    new_enh = '|'.join(sorted([dep_enh, head_enh], key=lambda x: int(x.split(':')[0])))
    parts[8] = new_enh
    new_line = '\t'.join(parts)
    return new_line


def add_conj_to_all(all_lines, dep_type):
    dep_lines = re.findall(r'.*\t{0}\t.*'.format(dep_type), all_lines)
    new_lines = all_lines
    for dep_line in dep_lines:
        new_lines = new_lines.replace(dep_line, add_conj_dep(dep_line, new_lines))
    return new_lines


# XCOMP
def change_nsubj_line(line, dep_num):
    parts = line.split('\t')
    enh_dep = parts[8]
    if '|' in enh_dep:
        enh_deps = enh_dep.split('|')
        enh_deps.append(dep_num + ':' + 'nsubj')
        new_enh_dep = '|'.join(sorted(enh_deps, key=lambda x: int(x.split(':')[0])))
    else:
        new_enh_dep = '|'.join(sorted([enh_dep, dep_num + ':' + 'nsubj'], key=lambda x: int(x.split(':')[0])))
    parts[8] = new_enh_dep
    new_line = '\t'.join(parts)
    return new_line


def add_xcomp_dep(dep_line, all_lines):
    parts = dep_line.split('\t')
    dep_num = parts[0]

    dep_head = parts[6]
    head_line = re.search(r'(\n|^){0}\t.*'.format(dep_head), all_lines).group()
    head_num = head_line.split('\t')[0]

    num_nsubj_str = '\t' + head_num + '\tnsubj'
    nsubj_line = re.search(r'(\n|^).*{0}(:pass)*\t.*?(\n|$)'.format(num_nsubj_str.strip()), all_lines)
    if nsubj_line:
        nsubj_line = nsubj_line.group()
        new_nsubj_line = change_nsubj_line(nsubj_line, dep_num)
        return nsubj_line, new_nsubj_line
    else:
        return nsubj_line, nsubj_line


def add_xcomp_to_all(all_lines):
    dep_lines = re.findall(r'.*\t{0}\t.*'.format('xcomp'), all_lines)
    new_lines = all_lines
    for dep_line in dep_lines:
        nsubj_line, new_nsubj_line = add_xcomp_dep(dep_line, all_lines)
        if nsubj_line:
            new_lines = new_lines.replace(nsubj_line, new_nsubj_line)
    return new_lines


def sort_deps(text_sent):
    text_sent = text_sent.strip()
    lines = text_sent.split('\n')
    for i in range(len(lines)):
        if lines[i][0] == "#":
            continue
        parts = lines[i].split('\t')
        enh_deps = parts[8].split('|')
        new_enh_dep = '|'.join(sorted(enh_deps, key=lambda x: int(x.split(':')[0])))
        parts[8] = new_enh_dep
        lines[i] = '\t'.join(parts)
    text_sent = '\n'.join(lines)
    return text_sent


with open('be_sents.conllu', 'r', encoding='utf-8') as f:
    text = f.read()
sents = text.split('\n\n')[:-1]

for i in range(len(sents)):
    lines = sents[i].split('\n')
    for j, line in enumerate(lines):
        if line[0] == '#':
            continue
        lines[j] = copy_basic_deps(line)
    sents[i] = '\n'.join(lines)
    sents[i] = add_case_to_all(sents[i], 'obl')
    sents[i] = add_case_to_all(sents[i], 'nmod')
    sents[i] = add_mark_to_all(sents[i], 'advcl')
    sents[i] = add_mark_to_all(sents[i], 'acl')
    sents[i] = add_conj_to_all(sents[i], 'conj')
    sents[i] = add_xcomp_to_all(sents[i])
    sents[i] = apply_acl_rule_to_sent(sents[i])
    sents[i] = apply_acl_rule_to_conj(sents[i])
    sents[i] = sort_deps(sents[i])

new_text = '\n\n'.join(sents) + '\n\n'
with open('be_sents_enh.conllu', 'w', encoding='utf-8', newline='\n') as f:
    f.write(new_text)
