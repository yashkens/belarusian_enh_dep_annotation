import re
from copy import deepcopy
from tqdm import tqdm


# split sentences into lines without meta
def get_lines(sent, ans_sent):
    meta_lines = re.search('^(#.*\n)*', sent).group()
    no_meta_sent = sent[len(meta_lines):]
    lines = no_meta_sent.split('\n')
    meta_lines_ans = re.search('^(#.*\n)*', ans_sent).group()
    no_meta_ans_sent = ans_sent[len(meta_lines_ans):]
    ans_lines = no_meta_ans_sent.split('\n')
    return lines, ans_lines, meta_lines


# match predicted token indices to one gold token index
def match_pred_tokens_to_gold(lines, ans_lines):
    ind_matches = {}
    i = 0
    num_of_skipped = 0
    for j in range(len(ans_lines)):
        # если равны, добавляем индекс-сопоставление, идем дальше
        try:
            if ans_lines[j].split('\t')[1] == lines[i].split('\t')[1]:
                ind_matches[ans_lines[j].split('\t')[0]] = [lines[i].split('\t')[0]]
                i += 1
                continue
        except:
            print('\n'.join(ans_lines))
            print('\n'.join(lines))
        # если не равны, но начинается с того же символа или если это [UNK], пропускаем
        # ЧТО ЭТО И ЗАЧЕМ?
        # if not ans_lines[j].split('\t')[1].startswith(lines[i].split('\t')[1]) and lines[i].split('\t')[1] != '[UNK]':
        #     num_of_skipped += 1
        #     print(ans_lines[j], lines[i])
        #     continue
        # в остальных случаях full_token - текущий токен (ответы), откусываем от него длину текущего токена (пред)
        full_token = ans_lines[j].split('\t')[1]
        full_token = full_token[len(lines[i].split('\t')[1]):]
        indices = [lines[i].split('\t')[0]]
        for line in lines[i+1:]:  # идем дальше по (пред) линиям - пока начала совпадают - пропускаем
            # !NEW! Added full_token != ''
            if full_token.startswith(line.split('\t')[1]) or line.split('\t')[1] == '[UNK]' and full_token != '':
                if line.split('\t')[1] != '[UNK]':
                    full_token = full_token[len(line.split('\t')[1]):]
                else:
                    full_token = full_token[1:]  # assumption that [UNK] is always only 1 symbol in length
                i += 1
                indices.append(line.split('\t')[0])
            else:
                break
        ind_matches[ans_lines[j].split('\t')[0]] = indices
        i += 1
    return ind_matches, num_of_skipped


# collect all tokens matched to one original-token in one list and in groups
def get_additional_lists(ind_matches):
    all_misplaced = []
    misplaced_groups = []
    for key, value in ind_matches.items():
        if len(value) > 1:
            all_misplaced.extend(value)
            misplaced_groups.append(value)
    return all_misplaced, misplaced_groups


# create our predict with fixed tokenization (only enh deps will be used from here)
def match_ud_tokenization(lines, all_misplaced, misplaced_groups):
    group_ind = 0
    new_lines = []
    i = 0
    while i < len(lines):
        if str(i+1) in all_misplaced:
            begin = misplaced_groups[group_ind][0]
            end = misplaced_groups[group_ind][-1]
            full_token = ''
            head_candidates = []
            for line in lines[int(begin)-1:int(end)]:
                i += 1
                dep_nums = [dep.split(':')[0] for dep in line.split('\t')[8].split('|')]
                for num in dep_nums:
                    if num not in misplaced_groups[group_ind] and line.split('\t')[8] not in head_candidates:
                        head_candidates.append(line.split('\t')[8])
                full_token += line.split('\t')[1]
            format_line = lines[int(end)-1]
            parts = format_line.split('\t')
            parts[1] = full_token
            parts[2] = full_token
            if len(head_candidates) == 0:
                head_candidates.append('0:X')
                print('no head candidates')
            parts[8] = head_candidates[0]  # simply chooses first candidate - place for improvement
            new_line = '\t'.join(parts)
            new_lines.append(new_line)
            group_ind += 1
        else:
            new_lines.append(lines[i])
            i += 1
    return new_lines


def fix_indices(new_lines):
    fixed_lines = []
    for i in range(len(new_lines)):
        parts = new_lines[i].split('\t')
        parts[0] = str(i+1)
        fixed_lines.append('\t'.join(parts))
    return fixed_lines


# fix head numbers in enh deps
def fix_dep_numbers(fixed_lines, ind_matches, num_of_skipped):
    new_deps_lines = []
    for line in fixed_lines:
        parts = line.split('\t')
        deps = parts[8].split('|')
        new_deps = []
        for dep in deps:
            num = dep.split(':')[0]
            new_num = num  # !!! NEW NUM MUST BE ALWAYS DEFINED - PROBLEM!!!
            if int(num) < 0:
                new_num = '0'
            for orig_ind in ind_matches:
                if num == '0' or int(num) < 1:
                    new_num = '0'
                    break
                if num in ind_matches[orig_ind]:
                    new_num = str(int(orig_ind) - num_of_skipped)
                    break
            new_deps.append(new_num + ':' + ':'.join(dep.split(':')[1:]))
        new_full_dep = '|'.join(list(set(new_deps)))  # SET: if they start to duplicate - collapse into one
        parts[8] = new_full_dep
        new_deps_lines.append('\t'.join(parts))
    return '\n'.join(new_deps_lines)


def match_if_original_shorter(lines, ans_lines):
    new_lines = deepcopy(lines)
    count_added = 0
    ind_matches = {}
    for i in range(len(lines)):
        try:
            if lines[i].split('\t')[1] == ans_lines[i + count_added].split('\t')[1] or lines[i].split('\t')[1] == '[UNK]':
                ind_matches[ans_lines[i + count_added].split('\t')[0]] = [lines[i].split('\t')[0]]
                continue
            if not ans_lines[i + count_added].split('\t')[1].startswith(lines[i].split('\t')[1]):
                j = 0
                for ans_line in ans_lines[i + count_added + 1:]:
                    if ans_line.split('\t')[1] == lines[i].split('\t')[1]:
                        parts = ans_lines[i + count_added +j].split('\t')
                        parts[8] = '0:X'
                        new_lines.insert(i, '\t'.join(parts))
                        count_added += 1
                        ind_matches[ans_lines[i + count_added].split('\t')[0]] = [parts[0]]
                        break
                    j += 1
            else:
                # print(lines)
                print('Something is wrong!')
        except:
            print('\n'.join(lines))
            exit()
    return new_lines, ind_matches


def align_tokenization(ans_sents, sents):
    sents_fixed_tokenization = []
    for i in tqdm(range(len(sents))):
        lines, ans_lines, meta_lines = get_lines(sents[i], ans_sents[i])
        # less in original
        ind_matches, num_skipped = match_pred_tokens_to_gold(lines, ans_lines)
        all_misplaced, misplaced_groups = get_additional_lists(ind_matches)
        new_lines = match_ud_tokenization(lines, all_misplaced, misplaced_groups)
        fixed_lines = fix_indices(new_lines)
        ud_sent = fix_dep_numbers(fixed_lines, ind_matches, num_skipped)
        # more in original
        new_lines, ind_matches = match_if_original_shorter(ud_sent.split('\n'), ans_lines)
        fixed_lines = fix_indices(new_lines)
        ud_sent = fix_dep_numbers(fixed_lines, ind_matches, 0)
        sents_fixed_tokenization.append(meta_lines + ud_sent)
    return sents_fixed_tokenization


def insert_enhanced_in_original(ans_file, predict_file, output_file):
    with open(ans_file, 'r', encoding='utf-8') as f:
        ans_text = f.read()
    with open(predict_file, 'r', encoding='utf-8') as f:
        text = f.read()
    ans_sents = ans_text.split('\n\n')[:-1]
    sents = text.split('\n\n')[:-1]
    new_sents = align_tokenization(ans_sents, sents)
    print('Alignment is done!\nInserting into original...')
    ans_with_pred_deps = []
    for i in tqdm(range(len(new_sents))):
        lines, ans_lines, meta = get_lines(new_sents[i], ans_sents[i])
        if len(lines) != len(ans_lines):
            print('Tokenization align error!')
        for j in range(len(lines)):
            parts = ans_lines[j].split('\t')
            parts[8] = lines[j].split('\t')[8]
            ans_lines[j] = '\t'.join(parts)
        ans_with_pred_deps.append(meta + '\n'.join(ans_lines))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(ans_with_pred_deps) + '\n\n')
