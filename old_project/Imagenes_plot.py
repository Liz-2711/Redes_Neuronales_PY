import matplotlib.pyplot as plt

class Visualizer:
    
    def __init__(self, show_images=False):
       
        self.show_images = show_images
    
    def show_single_image(self, image, label, prediction=None):
        
        if not self.show_images:
            return
        
        plt.figure(figsize=(3, 3))
        plt.imshow(image, cmap="gray")
        
        if prediction is not None:
            status = "✓" if prediction == label else "✗"
            plt.title(f"Real: {label} | Pred: {prediction} {status}")
        else:
            plt.title(f"Label: {label}")
        
        plt.axis("off")
        plt.show()
    
    def show_errors(self, errors, limit=20):
        
        errors_to_show = errors[:limit]
        
        if len(errors_to_show) == 0:
            print("\n¡No hubo errores! La red acertó todo.")
            return
        
        print(f"\n{'='*60}")
        print(f"PRIMEROS {len(errors_to_show)} ERRORES DE PREDICCIÓN")
        print(f"{'='*60}")
        
      
        n_cols = 4
        n_rows = (len(errors_to_show) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 3 * n_rows))
        
        for i, error in enumerate(errors_to_show):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(error["image"], cmap="gray")
            plt.title(f"Índice: {error['index']}\nPred: {error['prediction']} | Real: {error['label']}")
            plt.axis("off")
        
        plt.tight_layout()
        plt.show()
        
        # Imprimir detalles en consola
        print("\nDetalle de errores:")
        print(f"{'Índice':<10} {'Predicción':<12} {'Label Real':<12}")
        print("-" * 34)
        for error in errors_to_show:
            print(f"{error['index']:<10} {error['prediction']:<12} {error['label']:<12}")
    
    def show_results(self, summary):
      
        print(f"\n{'='*60}")
        print("RESULTADOS FINALES DE LA EVALUACIÓN")
        print(f"{'='*60}")
        print(f"Total de imágenes evaluadas: {summary['total']}")
        print(f"Aciertos:                    {summary['correct']}")
        print(f"Errores:                     {summary['incorrect']}")
        print(f"Precisión:                   {summary['accuracy_percentage']:.2f}%")
        print(f"{'='*60}")