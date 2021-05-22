import pandas as pd
from ufal.udpipe import Model, Pipeline, ProcessingError
from conllu import parse
import numpy as np
from copy import deepcopy
from scripts.bert_vectorizer_aligner import get_full_token_vectors, align_by_cosine_similarity
from tqdm import tqdm


def get_vectors_and_alignment(be_sent, ru_sent):
    be_token_words, be_token_vectors = get_full_token_vectors(be_sent)
    ru_token_words, ru_token_vectors = get_full_token_vectors(ru_sent)
    aligned_RU = align_by_cosine_similarity(ru_token_vectors, be_token_vectors, ru_token_words, be_token_words)
    aligned_BE = align_by_cosine_similarity(be_token_vectors, ru_token_vectors, be_token_words, ru_token_words)
    return be_token_words, ru_token_words, aligned_BE, aligned_RU


class Node:

    def __init__(self, index, token):
        self.index = index
        self.token = token
        self.be_index_pair = None
        self.be_token_pair = None
        self.edges = []
        if index == '0':
            self.be_index_pair = 0

    def add_edges(self, edges):
        self.edges.extend(edges)

    def transform_edge_to_be(self, nodes):
        self.be_edges = []
        for i in range(len(self.edges)):
            for node in nodes:
                if node.index == self.edges[i][0].index:
                    self.be_edges.extend([(node, self.edges[i][1])])
        if not self.be_edges:
            self.be_edges = [(Node('0', 'HEAD_NODE'), self.edges[i][1])]
        all_be_edges = [(self.be_edges[i][0].be_index_pair, self.be_edges[i][1]) for i in range(len(self.be_edges))]
        return all_be_edges


class SyntGraph:

    def add_edges_to_nodes(self):
        for i in range(len(self.nodes)):
            deps = self.parsed[0][i]['deps']
            if isinstance(deps, str):
                deps = deps.split('|')
                deps = [(':'.join(dep.split(':')[1:]), int(dep.split(':')[0])) for dep in deps]
            for dep in deps:
                index = dep[1]
                if index != 0:
                    index -= 1
                    self.nodes[i].add_edges([(self.nodes[index], dep[0])])
                else:
                    self.nodes[i].add_edges([(Node('0', 'HEAD_NODE'), dep[0])])

    def __init__(self, synt_ud_sent):
        self.parsed = parse(synt_ud_sent)
        self.nodes = [Node(word['id'], word['form']) for word in self.parsed[0]]
        self.add_edges_to_nodes()

    def add_be_indices(self, be_token_words, align_ru):
        used_indices = []
        for i, node in enumerate(self.nodes):
            node.be_token_pair = align_ru[i][1]
            temp_be_tokens = be_token_words
            if be_token_words.count(align_ru[i][1]) > 1:  # RU word appears more than once in BE words
                instances = []
                last_ind = 0
                while temp_be_tokens.count(align_ru[i][1]) != 0:
                    ind = temp_be_tokens.index(align_ru[i][1]) + last_ind
                    instances.append(ind)
                    last_ind = ind + 1
                    temp_be_tokens = temp_be_tokens[ind + 1:]
                dists = [abs(item - i) for item in instances]  # choose closest index
                node.be_index_pair = instances[np.argmin(dists)] + 1
                used_indices.append(node.be_index_pair)
            else:
                node.be_index_pair = be_token_words.index(align_ru[i][1]) + 1
                used_indices.append(node.be_index_pair)


def create_be_graph_from_ru_graph(enh_ru_sent, be_token_words, aligned_RU):
    graph = SyntGraph(enh_ru_sent)
    graph.add_be_indices(be_token_words, aligned_RU)
    be_graph = {}  # dict with UD indices and their edges
    for node in graph.nodes:
        if node.be_index_pair not in be_graph:
            be_graph[node.be_index_pair] = [node.transform_edge_to_be(graph.nodes)]
        else:
            be_graph[node.be_index_pair].append(node.transform_edge_to_be(graph.nodes))
    return be_graph, graph


def find_all_multiple_translations(be_graph, graph):
    multi_trans = []
    for ind, deps in be_graph.items():
        if len(deps) < 2:
            continue
        multi_pair = []
        multi_pair.append(ind)
        inds = []   # all indices of aligned in current index
        dep_heads = []  # all heads of deps in current index
        for node in graph.nodes:
            curr_dep_heads = []
            if node.be_index_pair == ind:
                inds.append(node.index)
                for edge in node.edges:
                    curr_dep_heads.append(edge[0].index)
                dep_heads.append(curr_dep_heads)
                multi_pair.append((node.token, node.be_token_pair, 'not_dep'))
        for i in range(len(inds)):
            for dep in dep_heads[i]:  # if head dep of current align is in indices of current align
                if dep in inds:       # it is dependent in this group and will not be used
                    multi_pair[i+1] = (multi_pair[i+1][0], multi_pair[i+1][1], 'is_dep')
        multi_trans.append(multi_pair)
    return multi_trans


def choose_most_similar(trans_pairs, aligned_RU):
    pair_weights = []
    for trans_trio in trans_pairs:
        if trans_trio[2] == 'is_dep':
            pair_weights.append(-100)
            continue
        for align_trio in aligned_RU:
            if trans_trio[0] == align_trio[0] and trans_trio[1] == align_trio[1]:
                pair_weights.append(align_trio[2])
    if len(pair_weights) > len(trans_pairs):
        pair_weights = pair_weights[:len(trans_pairs)]
    return trans_pairs[np.argmax(pair_weights)]


def create_disambiguated_be_tree(be_graph, graph, aligned_RU):
    be_graph_disamb = deepcopy(be_graph)
    multi_trans = find_all_multiple_translations(be_graph, graph)
    for trans in multi_trans:
        right_trans = choose_most_similar(trans[1:], aligned_RU)
        right_dep_ind = trans.index(right_trans) - 1  # -1 since the first item is index
        be_graph_disamb[trans[0]] = [be_graph[trans[0]][right_dep_ind]]
    return be_graph_disamb


def create_be_ud_with_enh_synt(be_ud_sent, be_graph_disamb, graph, be_base_synt, aligned_BE, be_token_words):
    parsed_be_ud = parse(be_ud_sent)
    for token in parsed_be_ud[0]:
        if token['id'] in be_graph_disamb:
            if token['deps'] != None:
                continue
            rev_deps = []
            for dep in be_graph_disamb[token['id']][0]:
                dep_str = (dep[1], dep[0])
                rev_deps.append(dep_str)
            token['deps'] = rev_deps
        else:  # some BE word did not occur in RU graph
            for pair in aligned_BE:
                if pair[0] == be_token_words[token['id']-1]:
                    for node in graph.nodes:
                        if node.token == pair[1]:
                            break  # current node will be used later
                    if token['id'] == be_base_synt[node.be_index_pair-1]['head']:  # found node is dependent
                        rev_deps = []
                        for dep in node.be_edges:
                            dep_str = (dep[1], dep[0].be_index_pair)
                            rev_deps.append(dep_str)
                        token['deps'] = rev_deps  # add deps to head of group
                        parsed_be_ud[0][node.be_index_pair-1]['deps'] = [('X', token['id'])]  # add X to dependent
                    elif node.be_index_pair == be_base_synt[token['id']-1]['head']:  # found node is head
                        token['deps'] = [('X', node.be_index_pair)]  # add X to dependent
                    else:  # no syntax relations in group - add 'X_dep'
                        token['deps'] = [('X_' + edge[1], edge[0].be_index_pair) for edge in node.be_edges]
    be_ud_synt = parsed_be_ud[0].serialize()
    return be_ud_synt


def create_ud_from_tokenized(ru_token_words):
    ud_lines = []
    format_line = '{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_'
    for i, word in enumerate(ru_token_words):
        line = format_line.format(str(i+1), word)
        ud_lines.append(line)
    ud_sent = '\n'.join(ud_lines)
    return ud_sent


def get_be_enh_sent(sent_ind, be_sents, ru_sents, enh_ru_sents):
    be_token_words, ru_token_words, aligned_BE, aligned_RU = get_vectors_and_alignment(be_sents[sent_ind], ru_sents[sent_ind])
    enh_ru_sent = enh_ru_sents[sent_ind]
    be_graph, graph = create_be_graph_from_ru_graph(enh_ru_sent, be_token_words, aligned_RU)
    be_graph_disamb = create_disambiguated_be_tree(be_graph, graph, aligned_RU)
    be_ud_sent = create_ud_from_tokenized(be_token_words)
    # add base syntax to ud BE sent
    be_base_synt_text = be_pipeline_parse.process(be_ud_sent, error)
    if error.occurred():
        print(error.message)
    be_base_synt = parse(be_base_synt_text)[0]
    be_enh_sent = create_be_ud_with_enh_synt(be_ud_sent, be_graph_disamb, graph, be_base_synt, aligned_BE, be_token_words)
    return be_enh_sent, be_base_synt_text


m = Model.load("models/belarusian-hse-ud-2.5.udpipe")
be_pipeline_parse = Pipeline(m, 'conllu', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
error = ProcessingError()


def align_russian_sents(translation_csv_file, enhanced_sents_file, output_file):
    df = pd.read_csv(translation_csv_file, index_col='Unnamed: 0')
    be_sents = df['original'].to_list()
    ru_sents = df['russian'].to_list()
    with open(enhanced_sents_file, 'r', encoding='utf-8') as f:
        enh_ru_text = f.read()
    enh_ru_sents = enh_ru_text.split('\n\n')[:-1]

    be_enhanced_sents = []
    be_base_sents = []
    for i in tqdm(range(len(be_sents))):
        be_enh, be_base_synt = get_be_enh_sent(i, be_sents, ru_sents, enh_ru_sents)
        be_enhanced_sents.append(be_enh)
        be_base_sents.append(be_base_synt)

    enh_with_base = []
    for i in tqdm(range(len(be_base_sents))):
        be_base_sents[i] = be_base_sents[i].strip()
        be_enhanced_sents[i] = be_enhanced_sents[i].strip()
        lines = be_base_sents[i].split('\n')
        enh_lines = be_enhanced_sents[i].split('\n')
        for j in range(len(lines)):
            parts = lines[j].split('\t')
            enh_parts = enh_lines[j].split('\t')
            parts[8] = enh_parts[8]
            lines[j] = '\t'.join(parts)
        enh_with_base.append('# text = ' + be_sents[i] + '\n' + '# ru_text = ' + ru_sents[i] + '\n' + '\n'.join(lines))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(enh_with_base) + '\n\n')


def align_ukrainian_sents(translation_csv_file, enhanced_sents_file, output_file):
    df = pd.read_csv(translation_csv_file, index_col='Unnamed: 0')
    be_sents = df['original'].to_list()
    ru_sents = df['ukrainian'].to_list()
    with open(enhanced_sents_file, 'r', encoding='utf-8') as f:
        enh_ru_text = f.read()
    enh_ru_sents = enh_ru_text.split('\n\n')[:-1]

    be_enhanced_sents = []
    be_base_sents = []
    for i in tqdm(range(len(be_sents))):
        be_enh, be_base_synt = get_be_enh_sent(i, be_sents, ru_sents, enh_ru_sents)
        be_enhanced_sents.append(be_enh)
        be_base_sents.append(be_base_synt)

    enh_with_base = []
    for i in range(len(be_base_sents)):
        be_base_sents[i] = be_base_sents[i].strip()
        be_enhanced_sents[i] = be_enhanced_sents[i].strip()
        lines = be_base_sents[i].split('\n')
        enh_lines = be_enhanced_sents[i].split('\n')
        for j in range(len(lines)):
            parts = lines[j].split('\t')
            enh_parts = enh_lines[j].split('\t')
            parts[8] = enh_parts[8]
            lines[j] = '\t'.join(parts)
        enh_with_base.append('# text = ' + be_sents[i] + '\n' + '# uk_text = ' + ru_sents[i] + '\n' + '\n'.join(lines))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(enh_with_base) + '\n\n')


# align_ukrainian_sents()
# align_russian_sents()
