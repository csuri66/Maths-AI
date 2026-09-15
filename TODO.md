Encoder: különböző problémákból tanulni edge embeddinget
- stable marriage és roommatet kap az encoder a megoldásokkal
- loss hozzá: Edge ranking loss: egy ágens jobb partnerét magasabbra értékelje, mint egy rosszabbat.

- Blocking-risk loss: egy edge környezetéből becsülje, hogy az adott részmatchingben blokkolóvá válna-e.
- Utility loss: tanuljon domain-specifikus vagy fairness-utility értéket.
- Contrastive loss: stabil megoldásban szereplő és nem szereplő, szerkezetileg hasonló élek szétválasztása.
- Decoder objective: a végső matchingre számolt stabilitási és optimalizálási reward.

Encoder:
- 3–4 edge-aware GATv2 layer
- hidden dim = 128
- 4 attention head
- residual + LayerNorm
- explicit edge state
- edge embedding dim = 128

Decoder:
- autoregresszív edge selector
- hard degree-1 action mask
- unmatched token minden ágenshez
- GRU-alapú state encoder
- greedy + beam search

Utófeldolgozás:
- exact blocking-pair detector
- local repair
- végső stable-validáció