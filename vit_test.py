import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, dim, channels=3):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size**2

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.Linear(patch_dim, dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.to_patch_embedding(x)

        # Add class token
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x += self.pos_embedding

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads, dim_head, mlp_dim, dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ViTDecoder(nn.Module):
    def __init__(self, dim, output_dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads, dim_head, mlp_dim, dropout))

        self.to_output = nn.Linear(dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        # Use the class token for the final output
        cls_token = x[:, 0]
        return self.to_output(cls_token)


class ViT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        channels=3,
        dropout=0.0,
        decoder_depth=1,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, dim, channels)
        self.encoder = ViTEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.decoder = ViTDecoder(
            dim, output_dim, decoder_depth, heads, dim_head, mlp_dim, dropout
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        return self.decoder(x)

    def replace_decoder(self, new_decoder):
        """Replace the decoder after training, e.g., for transfer learning."""
        self.decoder = new_decoder


def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device="cuda"
):
    """
    Train the ViT model

    Args:
        model: ViT model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on ('cuda' or 'cpu')

    Returns:
        Trained model
    """
    model = model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct.double() / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    return model


# Example usage
if __name__ == "__main__":
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    # Define hyperparameters
    image_size = 224
    patch_size = 16
    dim = 768
    depth = 12
    heads = 12
    dim_head = 64
    mlp_dim = 3072
    num_classes = 1000  # ImageNet classes

    # Create a ViT model for image classification
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        dim=dim,
        output_dim=num_classes,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        mlp_dim=mlp_dim,
        dropout=0.1,
        decoder_depth=1,
    )

    # Set up data loaders (simplified example)
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Assuming we have ImageNet data
    train_dataset = datasets.ImageFolder("path/to/imagenet/train", transform=transform)
    val_dataset = datasets.ImageFolder("path/to/imagenet/val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    print("Training the model for ImageNet classification...")
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=10
    )

    # Now let's adapt the model for a different task (e.g., sentiment analysis with 2 classes)
    print("Adapting the model for sentiment analysis...")

    # Create a new decoder for the new task
    new_output_dim = 2  # Positive/Negative sentiment
    new_decoder = ViTDecoder(
        dim=dim,
        output_dim=new_output_dim,
        depth=1,
        heads=heads,
        dim_head=dim_head,
        mlp_dim=mlp_dim,
        dropout=0.1,
    )

    # Replace the decoder
    model.replace_decoder(new_decoder)

    # Now we would fine-tune the model on sentiment data
    # (This is just a placeholder - in a real scenario you would load sentiment data)
    print("Model architecture updated for new task. Ready for fine-tuning.")

    # Example of freezing encoder weights and only training the new decoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Set up new optimizer that only updates the decoder parameters
    optimizer = optim.Adam(model.decoder.parameters(), lr=1e-4)

    # Fine-tune on new task (simplified example)
    print("Fine-tuning the model on the new task...")
    # train_model(model, sentiment_train_loader, sentiment_val_loader, criterion, optimizer, num_epochs=5)