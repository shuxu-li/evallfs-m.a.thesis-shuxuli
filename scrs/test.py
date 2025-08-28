def ask_model_batch(
    model,
    tokenizer,
    questions: dict,
    descriptions: pd.DataFrame,
    k: int = 2,
    target_list: list = None,
    scope_size: int = None,
    seed: int = 42,
    kw_context: bool = False,
    vl_propform: bool = False,
    justification: bool = False,
    repeat_num: int = 5,
    batch_id: str = None,
    batch_size: int = 10,
    model_name: str = "",
    max_new_tokens: int = None,
):
    import time
    if max_new_tokens is None:
        max_new_tokens = 256 if justification else 16

    model_short_name = model_name.split('/')[0]
    results_file, log_file, batch_id = setup_files(model_name, batch_id)
    set_seed(seed)
    results = []
    start_time_all = time.time()

    targets = target_list if target_list else list(questions.keys())

    for target in targets:
        # 每个 target 都打印一次 system prompt
        system_prompt = PROMPT_FIX + (AVEC_JSTF if justification else SIMPLE_RES)
        logging.info("\n================ SYSTEM PROMPT ===================")
        logging.info(f"Model: {model_short_name} | Kw_ctx: {int(kw_context)} | Vl_prop: {int(vl_propform)} | Justification: {int(justification)}")
        logging.info(f"System Prompt:\n{system_prompt}")
        logging.info("==================================================\n")

        for q_id in range(1, len(questions[target]) + 1):
            prompt, question = generate_prompt(
                questions=questions,
                descriptions=descriptions,
                question_id=q_id,
                scope_size=scope_size,
                k=k,
                target=target,
                kw_context=kw_context,
                vl_propform=vl_propform,
                justification=justification,
            )
            question_data = questions[target][q_id - 1]
            repeat_responses = []
            prompt_batch, metadata_batch = [], []

            for r in range(repeat_num):
                set_seed(seed + r)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt + "QUESTION:\n\n" + question},
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_batch.append(text)
                meta_info = question_data["ex_question"]
                metadata_batch.append((
                    target, q_id, k, r,
                    meta_info, question_data.get("scope", ""),
                    question_data.get("expected", "")
                ))

                if len(prompt_batch) >= batch_size or r == repeat_num - 1:
                    inputs = tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
                    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
                    generated_ids = [gen[len(inp):] for inp, gen in zip(inputs["input_ids"], generated)]
                    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                    for i, resp in enumerate(responses):
                        response_std = extract_response_std(resp)
                        meta = metadata_batch[i]
                        if response_std == "Unknown":
                            logging.warning(f"Unknown response for {model_short_name} on {target} Q{q_id} #{meta[3]+1}: \n{resp}")
                            continue
                        row = [
                            model_short_name, meta[0], meta[1], meta[5],
                            int(kw_context), int(vl_propform), int(justification),
                            meta[2], meta[3] + 1,
                            meta[4].get("keyword", ""), meta[4].get("value", ""),
                            meta[4].get("lf_name", ""), meta[6], response_std
                        ]
                        results.append(row)
                        repeat_responses.append((meta[3]+1, resp, response_std))

                    save_results(results, results_file)
                    prompt_batch, metadata_batch, results = [], [], []

            # 每个问题日志一次 + 所有 repeat response
            logging.info(f"---------- Target: {target} ----------")
            logging.info(f"> Q{q_id} (scope={question_data.get('scope', '')}):")
            logging.info(f"Prompt:\n{prompt}")
            logging.info(f"Question:\n{question}")
            logging.info(f"True LF: {question_data['ex_question'].get('lf_name', '')}")
            logging.info(f"Expected label: {question_data.get('expected', '')}")
            logging.info("-- Repeats:")
            for rid, rtext, rstd in repeat_responses:
                logging.info(f"[#{rid}] Response: {rtext} → {rstd}")
            logging.info("-------------------------------------------\n")

    elapsed = time.time() - start_time_all
    print(f"\nAll done in {elapsed:.2f}s")