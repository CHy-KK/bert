def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file,doc_stride,mode=None):
    writer = tf.python_io.TFRecordWriter(output_file)
    batch_tokens = []
    batch_labels = []
    batch_index = []
    feature_list_total = []
    # print(len(examples))
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature_list,ntokens_list,label_ids_list = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, doc_stride , mode)
        feature_list_total.extend(feature_list)
        for feature, ntokens, label_ids in zip(feature_list,ntokens_list,label_ids_list):
            batch_tokens.extend(ntokens)
            batch_labels.extend(label_ids)
            batch_index.extend([ex_index]*len(ntokens))
            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["mask"] = create_int_feature(feature.mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature(feature.label_ids)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
    # sentence token in each batch
    writer.close()
    return batch_tokens,batch_labels,batch_index,feature_list_total
