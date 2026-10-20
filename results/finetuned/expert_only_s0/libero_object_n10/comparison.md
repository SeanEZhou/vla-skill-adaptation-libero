| suite | task | instruction | baseline success (95% CI) | finetuned success (95% CI) | mean steps |
|---|---|---|---|---|---|
| libero_object | 0 | pick up the alphabet soup and place it in the basket | 5/5 = 100% (57%-100%) | 0/5 = 0% (0%-43%) | 211 |
| libero_object | 1 | pick up the cream cheese and place it in the basket | 5/5 = 100% (57%-100%) | 0/5 = 0% (0%-43%) | 204 |
| libero_object | 2 | pick up the salad dressing and place it in the basket | 4/5 = 80% (38%-96%) | 0/5 = 0% (0%-43%) | 213 |
| libero_object | 3 | pick up the bbq sauce and place it in the basket | 4/5 = 80% (38%-96%) | 0/5 = 0% (0%-43%) | 231 |
| libero_object | 4 | pick up the ketchup and place it in the basket | 5/5 = 100% (57%-100%) | 0/5 = 0% (0%-43%) | 210 |
| libero_object | 5 | pick up the tomato sauce and place it in the basket | 5/5 = 100% (57%-100%) | 0/5 = 0% (0%-43%) | 218 |
| libero_object | 6 | pick up the butter and place it in the basket | 5/5 = 100% (57%-100%) | 0/5 = 0% (0%-43%) | 212 |
| libero_object | 7 | pick up the milk and place it in the basket | 5/5 = 100% (57%-100%) | 0/5 = 0% (0%-43%) | 207 |
| libero_object | 8 | pick up the chocolate pudding and place it in the basket | 4/5 = 80% (38%-96%) | 0/5 = 0% (0%-43%) | 227 |
| libero_object | 9 | pick up the orange juice and place it in the basket | 4/5 = 80% (38%-96%) | 0/5 = 0% (0%-43%) | 225 |
| **all** | | | **46/50 = 92% (81%-97%)** |  | |
| **all** | | |  | **0/50 = 0% (0%-7%)** | |

**Paired (50 identical initial states):** baseline 92% -> finetuned 0%, diff -92% (bootstrap 95% CI -98% to -84%); discordant pairs: only-baseline 46, only-finetuned 0; exact McNemar p = 2.84e-14
