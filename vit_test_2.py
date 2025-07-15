import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """Convert images into patches and embed them.

    This module splits the input image into patches and linearly embeds each patch.
    The output is a sequence of embedded patches that can be processed by a Transformer.
    """

    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim),
        )

    def forward(self, x):
        return self.projection(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for capturing global dependencies.

    This allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        out = self.proj(out)
        return out


class MLP(nn.Module):
    """Multi-layer perceptron used in Transformer blocks.

    Consists of two linear layers with a GELU activation in between.
    """

    def __init__(self, embed_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP.

    Uses layer normalization and residual connections around both
    the attention and MLP blocks.
    """

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer Encoder.

    Processes image patches through a series of transformer blocks.
    Includes positional embeddings and a class token for classification tasks.
    """

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        num_heads=12,
        mlp_dim=3072,
        num_layers=12,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        return x


class MaskedViT(nn.Module):
    """Masked Vision Transformer for self-supervised learning.

    Implements a masked autoencoder approach where random patches are masked
    and the model is trained to reconstruct the missing patches.
    """

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        num_heads=12,
        mlp_dim=3072,
        num_layers=12,
        dropout=0.1,
        mask_ratio=0.75,
    ):
        super().__init__()
        self.encoder = ViTEncoder(
            img_size,
            patch_size,
            in_channels,
            embed_dim,
            num_heads,
            mlp_dim,
            num_layers,
            dropout,
        )
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, patch_size * patch_size * in_channels),
        )

    def random_masking(self, x):
        B, N, D = x.shape  # batch, sequence length (with cls token), dimension

        # Keep the cls token (first token)
        cls_token, x = x[:, :1, :], x[:, 1:, :]
        N = N - 1  # adjust sequence length after removing cls token

        # Generate random indices for masking
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

        # Sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # restore the original order

        # Keep the first len_keep elements
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Add back the cls token
        x_masked = torch.cat([cls_token, x_masked], dim=1)

        # Create mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask.scatter_(1, ids_keep, 0)

        return x_masked, mask, ids_restore

    def forward(self, x):
        # Encode
        x = self.encoder.patch_embed(x)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.encoder.pos_embed
        x = self.encoder.dropout(x)

        # Apply masking
        x_masked, mask, ids_restore = self.random_masking(x)

        # Pass through transformer blocks
        for block in self.encoder.transformer_blocks:
            x_masked = block(x_masked)

        x_masked = self.encoder.norm(x_masked)

        # Decode (only the cls token)
        x_decoded = self.decoder(x_masked[:, 1:, :])

        return x_decoded, mask, ids_restore

    def replace_decoder(self, new_decoder):
        """Replace the decoder part of the model with a new decoder."""
        self.decoder = new_decoder

    def load_weights(self, weights_path):
        """Load pre-trained weights from a file path."""
        self.load_state_dict(torch.load(weights_path))
        return self




def main():
    # Create a model
    model = MaskedViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        num_heads=12,
        mlp_dim=3072,
        num_layers=12,
        dropout=0.1,
        mask_ratio=0.75,
    )

    # Create a random input
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    reconstructed_patches, mask, ids_restore = model(x)
    print(f"Reconstructed patches shape: {reconstructed_patches.shape}")
    print(f"Mask shape: {mask.shape}")

    # Example of replacing the decoder after pretraining
    # For example, to use the model for classification
    new_decoder = nn.Sequential(
        nn.Linear(768, 1000),  # 1000 classes
    )
    model.replace_decoder(new_decoder)

    # Now the model can be used for classification
    x = torch.randn(2, 3, 224, 224)
    logits, _, _ = model(x)
    print(f"Classification logits shape: {logits.shape}")

    # Training loop example
    def train_masked_vit():
        """Example training loop for the Masked Vision Transformer."""
        # Hyperparameters
        batch_size = 64
        learning_rate = 1e-4
        weight_decay = 0.05
        epochs = 100

        # Initialize model
        model = MaskedViT(
            img_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            num_heads=12,
            mlp_dim=3072,
            num_layers=12,
            dropout=0.1,
            mask_ratio=0.75,
        )

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Create optimizer
        # AdamW is typically used for transformer models
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        # Example dataset and dataloader
        # In practice, you would use a real dataset like ImageNet
        # This is just a placeholder
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=1000):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                # Generate random images
                return torch.randn(3, 224, 224)

        dataset = DummyDataset()
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for batch_idx, images in enumerate(dataloader):
                images = images.to(device)

                # Forward pass
                # The model returns reconstructed patches, mask, and indices for restoration
                reconstructed_patches, mask, ids_restore = model(images)

                # Process original images to get ground truth patches
                with torch.no_grad():
                    # Extract patches from original images
                    patches = model.encoder.patch_embed.projection[0](
                        images
                    )  # Use the Rearrange operation

                    # Only compute loss on masked patches
                    # We need to restore the order of patches to match with the mask
                    B, N, D = patches.shape
                    patches = torch.gather(
                        patches, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D)
                    )

                    # Apply mask: only compute loss on masked patches
                    target = patches[mask.bool()].reshape(B, -1, D)

                # Compute reconstruction loss (Mean Squared Error)
                pred = reconstructed_patches
                loss = F.mse_loss(pred, target)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Print progress
                if batch_idx % 10 == 0:
                    print(
                        f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}"
                    )

            # Update learning rate
            scheduler.step()

            # Print epoch summary
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch: {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": avg_loss,
                    },
                    f"masked_vit_checkpoint_epoch_{epoch+1}.pt",
                )

        # Save final model
        torch.save(model.state_dict(), "masked_vit_pretrained.pt")
        print("Training completed!")

    def use_pretrained_model_for_classification(num_classes=1000):
        """Demonstrates how to use a pretrained MaskedViT model for image classification."""
        # Load the pretrained model
        pretrained_model = MaskedViT(
            img_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            num_heads=12,
            mlp_dim=3072,
            num_layers=12,
            dropout=0.1,
            mask_ratio=0.75,
        )

        # Load pretrained weights using the new method
        pretrained_model.load_weights("masked_vit_pretrained.pt")

        # Replace the decoder with a classification head
        classification_head = nn.Sequential(
            nn.Linear(768, num_classes),
        )
        pretrained_model.replace_decoder(classification_head)

        # Freeze the encoder parameters for transfer learning
        for param in pretrained_model.encoder.parameters():
            param.requires_grad = False

        # Example inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrained_model = pretrained_model.to(device)
        pretrained_model.eval()

        # Process a sample image
        sample_image = torch.randn(1, 3, 224, 224).to(device)
        logits, _, _ = pretrained_model(sample_image)
        print(f"Classification output shape: {logits.shape}")

    # Call the training function to train the model and save weights
    print("Starting model training...")
    train_masked_vit()
    print(
        "Training complete! Model weights have been saved to 'masked_vit_pretrained.pt'"
    )


# Example usage
if __name__ == "__main__":
    main()