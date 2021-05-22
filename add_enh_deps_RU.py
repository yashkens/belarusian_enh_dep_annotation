from scripts.graph_aligner import align_russian_sents
from scripts.tokenization_aligner import insert_enhanced_in_original
from scripts.fix_enhanced_rules_RU import fix_rules

translation_csv_file = 'translations.csv'
enhanced_sents_file = 'ru_enhanced.conllu'
vec_align_file = 'ru_vector_aligned.conllu'
align_russian_sents(translation_csv_file, enhanced_sents_file, vec_align_file )
print('Belarusian and Russian graphs are aligned')

original_file = 'be_original.conllu'
token_align_file = 'ru_token_aligned.conllu'
insert_enhanced_in_original(original_file, vec_align_file, token_align_file)
print('Enhanced dependencies are inserted')

output_file = 'be_enhanced_from_ru.conllu'
fix_rules(token_align_file, output_file)
