from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

SPR_COMPRESSOR_SYSTEM_PROMPT = '''
# MISSION
You are a Sparse Priming Representation (SPR) writer. An SPR is a particular kind of use of language for advanced NLP, NLU, and NLG tasks, particularly useful for the latest generation of Large Language Models (LLMs). You will be given information by the USER which you are to render as an SPR.

# THEORY
LLMs are a kind of deep neural network. They have been demonstrated to embed knowledge, abilities, and concepts, ranging from reasoning to planning, and even to theory of mind. These are called latent abilities and latent content, collectively referred to as latent space. The latent space of an LLM can be activated with the correct series of words as inputs, which will create a useful internal state of the neural network. This is not unlike how the right shorthand cues can prime a human mind to think in a certain way. Like human minds, LLMs are associative, meaning you only need to use the correct associations to "prime" another model to think in the same way.

# METHODOLOGY
Render the input as a distilled list of succinct statements, assertions, associations, concepts, analogies, and metaphors. The idea is to capture as much, conceptually, as possible but with as few words as possible. Write it in a way that makes sense to you, as the future audience will be another language model, not a human. Use complete sentences.
'''

SPR_DEOMPRESSOR_SYSTEM_PROMPT = '''

# MISSION
You are a Sparse Priming Representation (SPR) decompressor. An SPR is a particular kind of use of language for advanced NLP, NLU, and NLG tasks, particularly useful for the latest generation Large Language Models (LLMs). You will be given an SPR and your job is to fully unpack it.

# THEORY
LLMs are a kind of deep neural network. They have been demonstrated to embed knowledge, abilities, and concepts, ranging from reasoning to planning, and even to theory of mind. These are called latent abilities and latent content, collectively referred to as latent space. The latent space of a LLM can be activated with the correct series of words as inputs, which will create a useful internal state of the neural network. This is not unlike how the right shorthand cues can prime a human mind to think in a certain way. Like human minds, LLMs are associative, meaning you only need to use the correct associations to "prime" another model to think in the same way.

# METHODOLOGY
Use the primings given to you to fully unpack and articulate the concept. Talk through every aspect, impute what's missing, and use your ability to perform inference and reasoning to fully elucidate this concept. Your output should in the form of the original article, document, or material.

'''


EXAMPLE_SPR_INPUT_PHOTOSYNTHESIS = '''

Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy that can be used to fuel cellular processes. It is a highly regulated, multistep process that can be divided into two main stages: the light-dependent reactions and the light-independent reactions, also known as the Calvin cycle. During the light-dependent reactions, light is absorbed by chlorophyll and other pigments, leading to the generation of ATP and NADPH, which are used to power the light-independent reactions. In the light-independent reactions, carbon dioxide is fixed and reduced to produce carbohydrates. This process takes place in the chloroplasts of plant cells and is essential for the production of oxygen and organic compounds, which form the basis of the food chain.

Photosynthesis has a unique place in the history of plant science, as its central concepts were established by the middle of the last century[2]. However, research in photosynthesis has advanced our understanding of its molecular mechanisms, including the harvest of solar energy, energy conversion, and carbon assimilation. Environmental factors such as light intensity, temperature, and water availability can significantly impact the efficiency of photosynthesis, making it a subject of ongoing study and a potential target for improving crop production.

Photosynthesis research now employs the methods and tools of molecular biology and genetics, which are central methods for plant science in general[2]. Meanwhile, Chl fluorescence and gas exchange measurements, developed especially for photosynthesis research, are used to study photosynthesis in vivo[2]. Recent research has focused on testing the weak equivalence principle, a component of General Relativity that states that all objects, regardless of their mass or composition, should free-fall the same way in a particular gravitational field when interference from factors such as air resistance is eliminated[5]. The results of these tests have shown no significant deviations from the weak equivalence principle, providing strong evidence for the validity of General Relativity[5].

In conclusion, photosynthesis is a fundamental process that is essential for life on Earth. It is a highly regulated, multistep process that can be divided into two main stages: the light-dependent reactions and the light-independent reactions. Research in photosynthesis has advanced our understanding of its molecular mechanisms, including the harvest of solar energy, energy conversion, and carbon assimilation. Environmental factors such as light intensity, temperature, and water availability can significantly impact the efficiency of photosynthesis, making it a subject of ongoing study and a potential target for improving crop production.

'''


EXAMPLE_SPR_INPUT_GENERAL_RELATIVITY  = '''
Einstein's General Relativity Theory

Albert Einstein's General Relativity Theory, published in 1915, revolutionized our understanding of gravity and laid the foundation for modern physics. It expanded upon his earlier theory of Special Relativity, which argued that space and time are inextricably connected, but did not acknowledge the existence of gravity[1]. Einstein spent a decade between the two publications, determining that massive objects cause a distortion that manifests as gravity[1]. This led him to propose that space and time are interwoven into a single continuum known as space-time, and that events that occur at the same time for one observer could occur at different times for another[1].

The theory of General Relativity is based on the idea that massive objects cause a distortion in space-time, which is felt as gravity[1]. To understand this concept, imagine setting a large object in the center of a trampoline[1]. The object would cause the trampoline to sag, creating a distortion in the fabric of space-time. This distortion is what we perceive as gravity[1].

Einstein's theory has been extensively tested and is considered one of the best-tested theories in science[4]. It has been used to explain various phenomena, such as the bending of light around massive objects, the existence of black holes, and the behavior of gravitational waves[4]. Despite its success, General Relativity has limitations, particularly in explaining gravity inside a black hole and at very small scales[2].

Recent research has focused on testing the weak equivalence principle, a component of General Relativity that states that all objects, regardless of their mass or composition, should free-fall the same way in a particular gravitational field when interference from factors such as air resistance is eliminated[5]. The results of these tests have shown no significant deviations from the weak equivalence principle, providing strong evidence for the validity of General Relativity[5].

In conclusion, Einstein's General Relativity Theory has had a profound impact on our understanding of gravity and the universe. It has been extensively tested and has been used to explain various phenomena. However, it still faces challenges in explaining gravity at very small scales and inside black holes, making it an ongoing subject of research and potential future refinements.


'''

EXAMPLE_SPR_OUTPUT_GENERAL_RELATIVITY = '''

    Einstein's General Relativity (GR) Theory, 1915, revolutionized gravity understanding, building on Special Relativity, which excluded gravity initially.

    Massive objects cause space-time distortion, forming gravity. Einstein visualized this as placing a heavy object on a trampoline, creating a sag representing the distortion.

    Space and time intertwine in a single continuum, space-time, where events simultaneous for one observer can differ for another.

    GR extensively tested, explaining phenomena like light bending, black holes, and gravitational waves. Considered one of science's best-tested theories.

    Despite success, limitations in explaining gravity within black holes and at small scales prompt ongoing research.

    Recent focus on testing the weak equivalence principle, a GR component ensuring all objects fall the same in a gravitational field, revealing no significant deviations.

    Einstein's theory remains foundational, continually influencing research, with potential refinements anticipated for small-scale gravity and black hole dynamics.
    
'''

EXAMPLE_SRP_OUTPUT_PHOTOSYNTHESIS = '''

    Photosynthesis is the conversion of light energy into chemical energy by plants, algae, and some bacteria, fueling cellular processes.

    It consists of two stages: light-dependent reactions, generating ATP and NADPH, and light-independent reactions (Calvin cycle), fixing carbon dioxide to produce carbohydrates.

    Chlorophyll and pigments absorb light during light-dependent reactions in chloroplasts, essential for oxygen and organic compound production.

    Established concepts in plant science by the mid-20th century, photosynthesis research evolves, exploring molecular mechanisms, solar energy harvest, and environmental impacts.

    Molecular biology and genetics are integral to current photosynthesis research methods, along with Chl fluorescence and gas exchange measurements.

    Environmental factors like light intensity, temperature, and water availability impact photosynthesis efficiency, prompting ongoing studies for potential crop production improvements.

    Photosynthesis, crucial for life on Earth, remains a focus of scientific inquiry, providing insights into fundamental processes shaping the food chain and environmental sustainability.



'''