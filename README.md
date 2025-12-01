---



**Model Choice:**  
- We used **FaceNet (InceptionResnetV1)** to extract 512-dimensional face embeddings because it is pretrained on VGGFace2 and gives robust and discriminative features even for small datasets.  
- **Logistic Regression** was chosen as the classifier because it is simple, fast, and works well for a small number of classes and limited data.

**Preprocessing:**  
- Faces are detected and aligned using **MTCNN**, ensuring consistent input for the embedding model.  
- No heavy data augmentation was applied due to the small dataset size, but this can be added for future improvement.

**Performance:**  
- Works well on clear, frontal face images.  
- Example confidence values: 63.5%, 92.1%, 95.0%  
- Strength: Fast, simple, easy to reproduce.  
- Limitation: May fail on side faces, blurry, or very small images.  

**Conclusion:**  
- This project demonstrates a simple yet functional face recognition system capable of predicting actor names from images with confidence scores.  
- Can be improved with more images per actor and data augmentation for better generalization.
