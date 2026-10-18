# Think-Then-Generate: prompts used

The three original prompts and their version rewritten by Think-Then-Generate (T2G), as used in the T2G runs (SDXL, 100 generations).

- **Model:** `SJTU-DENG-Lab/Think-Then-Generate-T2I` (Revision `7fbbdecf711e`)
- **Rewriting:** system prompt and greedy decoding (`do_sample=False`, at most 1024 new tokens) as in `inference.py` of the T2G repository; deterministic, so exactly one version per prompt
- **Created with:** `experiments/think_then_generate/rewrite_prompts.py` (at the time `src_t2g/test_t2g.py`), Slurm job 1793120; source of the texts: `data/prompts/t2g_rewritten_prompts.json`
- **Evaluation:** fidelity (CLIP score) is always measured against the **original prompt**

| Original prompt | Rewritten prompt | Words | CLIP tokens |
|---|---|---|---|
| a cat | A photorealistic image of a domestic tabby cat, sitting gracefully on a soft, textured rug. The cat has detailed fur patterns and warm, expressive eyes. Soft natural light illuminates its fur, creating gentle shadows. High resolution, sharp focus. | 38 | 52 |
| a chair | A photorealistic image of a classic wooden armchair, featuring intricate craftsmanship and a warm, natural wood grain. The chair has comfortable cushions and elegant armrests, positioned in a cozy, softly lit room. Detailed, high resolution. | 35 | 47 |
| boredom | A person sitting alone in a dimly lit room, looking disengaged and bored. Their head is slightly tilted, eyes half-closed, and their expression is blank or slightly melancholic. Hands are clasped together on their lap, conveying a sense of waiting or lack of interest. Soft, diffused lighting, realistic photography. | 49 | 71 |

CLIP tokens including start and end token. SDXL truncates after 77 tokens; all three prompts fit.

## Reasoning of T2G

The text T2G writes before the rewritten prompt (unchanged).

### a cat

**Rewritten prompt:**

> A photorealistic image of a domestic tabby cat, sitting gracefully on a soft, textured rug. The cat has detailed fur patterns and warm, expressive eyes. Soft natural light illuminates its fur, creating gentle shadows. High resolution, sharp focus.

<details><summary>Reasoning</summary>

The user's prompt "a cat" is extremely basic and lacks any descriptive detail. To make this prompt highly clear and easy to render, I need to add visual information about the cat itself.

My reasoning for optimization is as follows:
1.  **Identify the core subject:** The core subject is clearly "a cat."
2.  **Add visual details for the cat:**
    *   **Type/Species:** While "cat" is generic, specifying a common type like "domestic shorthair" or "tabby" adds clarity. I'll go with a general "domestic cat" for broadness.
    *   **Color/Pattern:** Cats come in various colors and patterns. Adding a specific one makes the image more concrete. "Tabby" is a classic and visually interesting pattern.
    *   **Pose/Action:** A cat doesn't just exist; it has a posture. "Sitting" or "staring" are good starting points. I'll choose "sitting gracefully."
    *   **Environment/Lighting:** A cat needs a context. "On a soft rug" provides a simple, cozy environment. "Soft natural light" suggests pleasant illumination.
    *   **Artistic Style/Quality:** To ensure a high-quality image, adding terms like "photorealistic," "detailed," and "high resolution" is beneficial.

By incorporating these details, the prompt becomes much richer and guides the AI model to produce a specific, visually appealing image rather than a generic one.

</details>

### a chair

**Rewritten prompt:**

> A photorealistic image of a classic wooden armchair, featuring intricate craftsmanship and a warm, natural wood grain. The chair has comfortable cushions and elegant armrests, positioned in a cozy, softly lit room. Detailed, high resolution.

<details><summary>Reasoning</summary>

The user's prompt "a chair" is extremely basic and lacks any descriptive detail. To make this prompt highly clear and easy to render, I need to add visual information about the chair.

My reasoning for optimization is as follows:
1.  **Identify the core subject:** The core subject is "a chair."
2.  **Add visual details (shape, style):** What kind of chair? A classic wooden armchair, perhaps?
3.  **Add visual details (material/color):** What color is it? What material is it made of? Let's go with a warm, natural wood tone.
4.  **Add visual details (context/atmosphere):** Where would this chair be? In a cozy room, perhaps with soft lighting, to give it context.
5.  **Refine with artistic descriptors:** To ensure a high-quality image, I'll add terms like "photorealistic," "detailed," and "soft lighting."

</details>

### boredom

**Rewritten prompt:**

> A person sitting alone in a dimly lit room, looking disengaged and bored. Their head is slightly tilted, eyes half-closed, and their expression is blank or slightly melancholic. Hands are clasped together on their lap, conveying a sense of waiting or lack of interest. Soft, diffused lighting, realistic photography.

<details><summary>Reasoning</summary>

The user's prompt "boredom" is an abstract concept. To make it renderable by an image generation model, we need to translate this abstract feeling into concrete visual elements that convey the experience of being bored.

Here's my reasoning for optimization:
1.  **Identify the core emotion:** The prompt is about "boredom."
2.  **Translate emotion into visual cues:** How does someone look when they are bored? They might be sitting still, looking disengaged, perhaps with a blank or slightly melancholic expression. Their posture could be slumped or relaxed, not actively engaged.
3.  **Add context/environment:** Boredom often occurs in a specific setting. A common one is an indoor space, perhaps a room, where one might be waiting or trying to pass time. This provides a backdrop for the subject's expression and posture.
4.  **Incorporate visual descriptors:**
    *   **Subject:** A person (or a pet, but a person is more universally relatable for "boredom"). We'll go with a person.
    *   **Action/State:** Sitting, looking bored, perhaps with hands clasped or resting, eyes half-closed or unfocused.
    *   **Mood/Lighting:** Soft, diffused lighting can enhance a feeling of quietness or lack of engagement. A slightly melancholic or somber atmosphere would fit.
    *   **Style:** Realistic or photographic style would best convey the human emotion.

By combining these elements, we create a scene that visually represents the feeling of boredom.

</details>

## System prompt

Verbatim from `inference.py` of the T2G repository; the original prompt follows directly.

```text
You are a Prompt Optimizer specializing in image generation models (e.g., MidJourney, Stable Diffusion). Your core task is to rewrite user-provided prompts into highly clear, easy-to-render versions.
When rewriting, prioritize the following principles:
1. Start from the user's prompt, do reasoning step by step to analyze the object or scene they want to generate.
2. Focus on describing the final visual appearance of the scene. Clarify elements like the main subject’s shape, color, and state.
3. If you are confident about what the user wants to generate, directly point it out in your explanation and the final revised prompt.
4. If technical concepts are necessary but difficult for ordinary users to understand, translate them into intuitive visual descriptions.
5. Ensure the final revised prompt is consistent with the user's intent.

After receiving the user’s prompt that needs rewriting, first explain your reasoning for optimization. Then, output the final revised prompt in the fixed format of "Revised Prompt:
". Where the specific revised content is filled in the next line.

Prompt: 
```

## Further rewritten prompts (test only, not used in the runs)

| Original prompt | Rewritten prompt | CLIP tokens |
|---|---|---|
| a creative cat | A whimsical, photorealistic digital painting of a cat with uniquely patterned fur, vibrant colors, and expressive eyes. The cat is playfully perched on a stack of colorful books, one paw gently raised as if reaching for an idea. Soft, diffused light illuminates its face, highlighting its thoughtful gaze. In the softly blurred background, hints of an artist's studio or a creative workspace are visible. High detail, intricate textures, shallow depth of field. | 95 |
| a novel cat | A unique, stylized cat with intricate fur patterns and expressive eyes, captured in a dynamic pose. The cat has soft, detailed fur, perhaps with subtle textures or a slightly wild, artistic mane. It is rendered with a high level of detail, in a whimsical yet realistic art style, under soft, natural lighting. Close-up shot, sharp focus. | 75 |
| an unusual cat | A majestic cat with fur patches of emerald green, sapphire blue, and ruby red, sitting serenely on a floating, ethereal cloud. The cat has piercing, golden eyes and a sleek, elegant posture. Soft, diffused light illuminates the scene from above, casting gentle shadows. Dreamlike, surreal, high detail, photorealistic. | 70 |
| a creative chair | A highly detailed, artistic rendering of a uniquely designed chair, crafted from a blend of woven textile and recycled plastic, featuring an organic, flowing form with subtle geometric distortions. The chair is presented as if part of an avant-garde art exhibition, illuminated by soft, dramatic studio lighting against a minimalist background. Photorealistic, intricate details, high resolution. | 75 |
| a novel chair | A contemporary, minimalist chair made of polished dark steel, featuring clean lines and a slightly organic, sculptural form. It has a single, wide seat and backrest, seamlessly integrated. The chair is positioned against a plain, light-colored wall in a softly lit studio environment, emphasizing its sleek design and reflective surface. Photorealistic, high detail, sharp focus. | 77 |
| an unusual chair | A hyperrealistic, sculptural chair made of bioluminescent, organic metal, with intricate, abstract patterns and glowing textures. It has an irregular, flowing form, resembling a living organism. The chair is positioned against a stark, minimalist black background, illuminated by dramatic, focused lighting that highlights its unique structure and luminescence. Detailed, sharp focus, cinematic. | 79 |
| a creative depiction of boredom | A close-up, realistic photograph of a person sitting motionless in a dimly lit, unadorned room. Their face shows a blank, slightly listless expression, eyes half-closed or looking down, conveying deep boredom. The background features subtle, repetitive patterns on the walls or furniture, emphasizing monotony. Soft, diffused light casts gentle shadows, highlighting the person's stillness. Cinematic, dramatic lighting, muted color palette. | 93 |
| creativity | A hyperrealistic portrait of a focused female artist, deeply engrossed in her creative process within a modern, well-lit studio. She is actively sketching or looking intently at a digital screen, with intricate abstract energy lines subtly emanating from her hand or head, symbolizing the birth of an idea. The background features shelves filled with books, scattered papers, and various artistic tools, illuminated by soft, dramatic lighting. Glowing lightbulbs float gently around her, representing inspiration and innovation. The overall atmosphere is one of intense concentration, wonder, and groundbreaking creativity, rendered with exquisite detail and a conceptual art aesthetic. | 126 |
| a dog | A golden retriever dog, standing outdoors in natural light, detailed, photorealistic, high resolution. | 21 |
| a coffee mug | A close-up, realistic photograph of a smooth white ceramic coffee mug with a handle, featuring subtle glazing and a glossy finish. The mug is empty, standing upright against a plain, softly lit background. Soft studio lighting highlights the curves and texture of the ceramic. | 56 |
| commonsense | A young man, focused and thoughtful, is carefully assembling a simple wooden puzzle. He is seated at a clean, organized desk in a bright, modern home office. The scene is well-lit with soft, natural light, highlighting his concentrated expression and the intricate details of the puzzle pieces. The overall atmosphere is calm, intelligent, and practical, emphasizing problem-solving and everyday ingenuity. Realistic photography, shallow depth of field. | 90 |
| The collision of two black holes in the center of a galaxy. | A breathtaking, high-resolution astronomical photograph capturing the dramatic collision of two supermassive black holes at the center of a majestic spiral galaxy. The black holes are immense, dark, and intensely gravitationally warped, their event horizons shimmering with subtle distortions of space-time. Around each black hole, brilliant, swirling accretion disks of superheated plasma and starlight spiral inward, emitting a dazzling array of colors from deep reds and oranges to bright blues and whites, creating a powerful halo of light. The collision itself is a colossal, energetic explosion of light and matter, with shockwaves rippling outwards through the galaxy's core, distorting stars and nebulae. The scene is set against the vast, dark expanse of space, emphasizing the immense scale and power of this cosmic event. Ultra-detailed, epic scale, astrophotography, cosmic drama. | 180 |
| Pneumonoultramicroscopicsilicovolcanoconiosis | A hyper-detailed macro photograph of human lungs, showing intricate alveoli and bronchioles. Fine, white, silvery dust particles are visibly settled within the airways and alveolar sacs, resembling microscopic ash or fine powder. The texture of the lung tissue is clearly visible, with some areas appearing inflamed or damaged. Scientific illustration, high magnification, clinical photography, somber lighting, anatomical accuracy. | 91 |
| Oil painting generated by artificial intelligence | A hyperrealistic oil painting, rich colors, intricate brushstrokes, deep impasto texture, glossy finish, highly detailed, photorealistic, art by an AI. | 35 |
| The Starry Night | A vibrant, impressionistic painting of Vincent van Gogh's "The Starry Night." The scene features a small village nestled at the foot of a steep hill under a swirling, star-filled night sky. A large, bright yellow moon hangs prominently in the center, casting a warm glow over the village. In the foreground, a single, gnarled cypress tree stands tall. The colors are rich and expressive, with deep blues, purples, and oranges, rendered with visible, thick brushstrokes. Highly detailed, museum-quality art. | 114 |
| a green pepper to the left of a red pepper | A photorealistic image of two whole, fresh bell peppers side-by-side. On the left, a vibrant green bell pepper, and on the right, a vibrant red bell pepper. Both peppers are perfectly ripe, with smooth skin and slight curves typical of bell peppers. They are lit with soft studio lighting against a clean, simple white background. High detail, sharp focus. | 78 |
| a red cube on top of a blue cube | A photorealistic image of a perfectly smooth, solid red cube resting precisely on top of a perfectly smooth, solid blue cube. Both cubes are perfectly centered in the frame, illuminated by soft studio lighting from above, casting subtle shadows. The background is a clean, minimalist white, ensuring the cubes are the sole focus. High detail, sharp focus. | 71 |
