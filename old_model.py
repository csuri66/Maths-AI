"""
model = GAT.GATEdgeClassifier(train_data[0].x.size(-1), 16)
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.015)


patience = 10
min_delta = 1e-4
eval_every = 10

best_val_loss = float("inf")
global_step = 0
stop_training = False


if training:
    train(global_step, best_val_loss, stop_training, 0.2, 0.3, 0.5,model,optimizer,train_data, group_size, eval_every, val_data, patience, min_delta)
model.load_state_dict(torch.load("best_model.pt", weights_only=True))



model.eval()


good=0
bad=0
if GAT_TEST:
    with torch.no_grad():
        for i in range(2500):
            pair_dict = {}

            group_size=3
            acc_graph = data_generator.generate_graph_m(group_size)

            structure = random.randint(0,100)
            num = structure %2
            match num:
                case 0:
                    acc_data = data_generator.graph_to_pyg_data_low_diff(acc_graph, group_size)
                case 1:
                    acc_data = data_generator.graph_to_pyg_data_high_diff(acc_graph, group_size)

            acc_data.x = (acc_data.x - mean) / std
            acc_data.edge_attr = (acc_data.edge_attr - mean_edge) / std_edge

            logits = model(acc_data)
            probs = torch.sigmoid(logits)
            propr_pref=acc_data.proposer_pref
            prope_pref = acc_data.proposee_pref
            ag = propr_pref |prope_pref
            original_data = acc_data
            if not hasattr(original_data, "orig_id") or original_data.orig_id is None:
                original_data.orig_id = torch.arange(
                    original_data.x.size(0),
                    device=original_data.x.device
                )

            remaining_orig_ids = original_data.orig_id.clone()

            for i in range(group_size - 1):

                current_data = build_current_data_from_original(
                    original_data,
                    remaining_orig_ids
                )

                if i == 0:
                    current_probs = probs
                else:
                    logits = model(current_data)
                    current_probs = torch.sigmoid(logits)
                best_e, (best_src, best_dst) = best_pairing_for_selected_node(
                    selected_nodes=range(current_data.x.size(0)),
                    edge_index=current_data.edge_index,
                    edge_probs=current_probs,
                    preference_lists=ag,
                    tolerance=0.01
                )
                if best_e is None:
                    break

                orig_u, orig_v = current_data.orig_id[[best_src, best_dst]].tolist()
                pair_dict[orig_u] = orig_v

                remaining_orig_ids = remaining_orig_ids[
                    (remaining_orig_ids != orig_u) & (remaining_orig_ids != orig_v)
                    ]

                if acc_graph.has_node(orig_u):
                    acc_graph.remove_node(orig_u)
                if acc_graph.has_node(orig_v):
                    acc_graph.remove_node(orig_v)

            if remaining_orig_ids.numel() == 2:
                u, v = remaining_orig_ids.tolist()
                pair_dict[u] = v

            if gale_shapley.is_stable_matching(pair_dict, propr_pref,  prope_pref):
                good += 1
            else:
                bad += 1

    print("GRAPH LEVEL ACCURACY GATv2 TEST")
    print(f"Accuracy: {(good/(good+bad))*100} %")


if roommate:
    print("Stable roommate case")
    g_size = 4
    graph_r = data_generator.generate_graph_r(g_size)

    data_r = data_generator.graph_to_pyg_data_r_random(graph_r,g_size)
    print(data_r.prefs)
    print(data_r.x)
    print(data_r.edge_attr)
    data_r.x = (data_r.x - mean) / std
    data_r.edge_attr = (data_r.edge_attr - mean_edge) / std_edge
    logits = model(data_r)
    probs = torch.sigmoid(logits)
    print(probs)

if LLM_FILE_GEN:
    with open("LLM_PREF_DATA_M.txt", "w", encoding="utf-8") as f:
        for i in range(100):
            low_diff = data_generator.graph_to_pyg_data_low_diff(data_generator.generate_graph_m(3), 3)
            f.write(str(low_diff.proposer_pref)+";")
            f.write(str(low_diff.proposee_pref)+ "\n")
            high_diff=  data_generator.graph_to_pyg_data_high_diff(data_generator.generate_graph_m(3), 3)
            f.write(str(high_diff.proposer_pref)+";")
            f.write(str(high_diff.proposee_pref)+ "\n")

good = 0
bad = 0
if LLM_TEST:
    proposer_prefs = []
    proposee_prefs = []
    with open("LLM_PREF_DATA_M.txt", "r", encoding="utf-8") as f:
        for line in f:
            prefs = line.strip().split(";")
            proposer_prefs.append(ast.literal_eval(prefs[0]))
            proposee_prefs.append(ast.literal_eval(prefs[1]))
    index = 0
    with open("stable_matchings_only.txt", "r", encoding="utf-8") as f:
        for line in f:
            result = ast.literal_eval(line)
            if gale_shapley.is_stable_matching(result, proposer_prefs[index], proposee_prefs[index]):
                good +=1
            else:
                bad +=1
            index += 1
    print("GRAPH LEVEL ACCURACY LLM TEST")
    print(good/(good+bad))


if LOCAL_LLM:
    import requests

    prompts = []
    with open("prompts_for_local_llm.txt", "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(line)

    for i, prompt in enumerate(prompts, 1):
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
        )
        data = r.json()
        print(f"\n--- Prompt {i} ---")
        print(prompt)
        print("\n--- Válasz ---")
        print(data["response"])
"""