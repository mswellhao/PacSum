from pyrouge import Rouge155
import os, shutil, random, string

from gensim_preprocess import preprocess_documents


def evaluate_rouge(summaries, references, remove_temp=False, rouge_args=[]):
    '''
    Args:
        summaries: [[sentence]]. Each summary is a list of strings (sentences)
        references: [[[sentence]]]. Each reference is a list of candidate summaries.
        remove_temp: bool. Whether to remove the temporary files created during evaluation.
        rouge_args: [string]. A list of arguments to pass to the ROUGE CLI.
    '''
    temp_dir = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    temp_dir = os.path.join("temp",temp_dir)
    print(temp_dir)
    system_dir = os.path.join(temp_dir, 'system')
    model_dir = os.path.join(temp_dir, 'model')
    # directory for generated summaries
    os.makedirs(system_dir)
    # directory for reference summaries
    os.makedirs(model_dir)
    print(temp_dir, system_dir, model_dir)

    assert len(summaries) == len(references)
    for i, (summary, candidates) in enumerate(zip(summaries, references)):
        summary_fn = '%i.txt' % i
        for j, candidate in enumerate(candidates):
            candidate_fn = '%i.%i.txt' % (i, j)
            with open(os.path.join(model_dir, candidate_fn), 'w') as f:
                #print(candidate)
                f.write('\n'.join(candidate))

        with open(os.path.join(system_dir, summary_fn), 'w') as f:
            f.write('\n'.join(summary))

    args_str = ' '.join(map(str, rouge_args))
    rouge = Rouge155(rouge_args=args_str)
    rouge.system_dir = system_dir
    rouge.model_dir = model_dir
    rouge.system_filename_pattern = '(\d+).txt'
    rouge.model_filename_pattern = '#ID#.\d+.txt'

    #rouge_args = '-c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a'
    #output = rouge.convert_and_evaluate(rouge_args=rouge_args)
    output = rouge.convert_and_evaluate()

    r = rouge.output_to_dict(output)
    print(output)
    #print(r)

    # remove the created temporary files
    #if remove_temp:
    #    shutil.rmtree(temp_dir)
    return r

def clean_text_by_sentences(text):
    """Tokenize a given text into sentences, applying filters and lemmatize them.

    Parameters
    ----------
    text : str
        Given text.

    Returns
    -------
    list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Sentences of the given text.

    """
    original_sentences = text
    filtered_sentences = [join_words(sentence) for sentence in preprocess_documents(original_sentences)]

    return filtered_sentences


def join_words(words, separator=" "):
    """Concatenates `words` with `separator` between elements.

    Parameters
    ----------
    words : list of str
        Given words.
    separator : str, optional
        The separator between elements.

    Returns
    -------
    str
        String of merged words with separator between elements.

    """
    return separator.join(words)
