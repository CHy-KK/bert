def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, doc_stride ,mode):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature

    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [I-PER,I-PER,X,O,O,O,X]

    """
    label_map = {}
    #here start with zero this means that "[PAD]" is zero
    for (i,label) in enumerate(label_list):
        label_map[label] = i
    with open(FLAGS.middle_output+"/label2id.pkl",'wb') as w:
        pickle.dump(label_map,w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i,(word,label) in enumerate(zip(textlist,labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i,_ in enumerate(token):
            if i==0:
                labels.append(label)
            else:
                labels.append("X")
    # print(">>>>>>>>>>>>>>>>", len(tokens))
    max_tokens_for_doc = max_seq_length - 1
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(tokens):
        length = len(tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(tokens):
            break
        start_offset += min(length, doc_stride)

    feature_list, ntokens_list, label_ids_list = [], [], []
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        # token_to_orig_map = {}
        token_is_max_context = {}
        ntokens = []
        label_ids = []
        segment_ids = []
        ntokens.append("[CLS]")
        label_ids.append(label_map["[CLS]"])
        segment_ids.append(0)
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            # token_to_orig_map[len(ntokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[split_token_index] = is_max_context
            ntokens.append(tokens[split_token_index])
            label_ids.append(label_map[labels[split_token_index]])
            segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            ntokens.append("[PAD]")
        # print(len(input_ids))
        assert len(input_ids) == max_seq_length
        assert len(mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(ntokens) == max_seq_length

        feature = InputFeatures(
            input_ids=input_ids,
            mask=mask,
            segment_ids=segment_ids,
            label_ids=label_ids,
            token_is_max_context=token_is_max_context,
            doc_span_index=doc_span_index,
            example_index=ex_index
        )
        feature_list.append(feature)
        ntokens_list.append(ntokens)
        label_ids_list.append(label_ids)

    if ex_index < 3:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        # print(tokens)
        logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    return feature_list,ntokens_list,label_ids_list
