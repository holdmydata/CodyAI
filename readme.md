# CodyAI - Learning in Process

A repository storing my learnings and work on GPT models, AI architecture, and other happenings as I try to figure all this stuff out. :P  

## Current Work
##### *The beginnings of a GPT model:*
![It's sentient...](https://cdn.discordapp.com/attachments/707638742080553061/1099771552435802225/image.png)
- 4/23/2023 
    - Added validation to the model (very barebone)
    - Preprocessing adjustments including removing urls and other unnecessary characters. Needs to be cleaned up and feed into the model. 
[CodyAI](https://github.com/holdmydata/CodyAI) - Writing a GPT model from scratch. Using [Andrej Karpathy's YouTube tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY), will build off of this as I work toward a personal AI model. Notes added for my own learning.
- 5/5/2023
    - A lot of preprocessing study and work. Next task is to remove reptitive punctuation and other characters.
    - Preprocessing: I combined all text into a single corpus and tried training on that. The graph is pretty, at least. Still, not working.
    - Preprocessing ToDo: Begin researching instruction/prompt generation, and possible sentiment analysis on user input.
    - Model: Checkpoints! I'm saving checkpoints now. I'm also saving the model, but I'm not sure if it's working. I'll have to test it out.
    - Model: Added Eval option in parameters to allow quick test of input. 
    - Model ToDo: A TON. 
 [5/5 First Combined Model Run. Forgot to change the training to a new model...so it failed...](https://github.com/holdmydata/CodyAI/blob/master/GPT/assets/First_corpus_run_combined_sentences.PNG)
 
## Next Steps
1. Understanding the encoder architecture, learn how it works, and add to model.
2. Research and build backend memory for the model. Vectorize dimensional databases. 
3. Panel/UI package for model training and evaluation.

## License

[MIT](https://choosealicense.com/licenses/mit/) 