from model import TarnetBase, EarlyStopper, outcome_loss, QiniEarlyStopper
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from metrics import auqc
import torch 
import numpy as np
import copy

class Tarnet:
    def __init__(
        self, 
        input_dim,
        shared_hidden=200, 
        outcome_hidden=100, 
        epochs=70,
        learning_rate= 1e-3,
        weight_decay = 1e-5,
        early_stop_metric='qini',
        use_ema=True,
        ema_alpha=0.15,
        patience=20,
        early_stop_start_epoch=0,
        shared_dropout = 0,
        outcome_droupout = 0,
        activation = torch.nn.ReLU
    ):
        self.model = TarnetBase(input_dim,shared_hidden=shared_hidden, outcome_hidden=outcome_hidden, shared_dropout=shared_dropout, outcome_dropout=outcome_droupout, activation=activation)
        self.epoch = epochs
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.early_stop_metric = early_stop_metric      
        # EMA parameters
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha
        self.patience = patience
        self.early_stop_start_epoch = early_stop_start_epoch
        
        # Tracking cho best model dựa trên Qini score
        self.best_qini = -np.inf
        self.best_loss = np.inf
        self.best_epoch = 0
        self.best_model_state = None
        
        # EMA tracking
        self.ema_qini = None
        self.best_ema_qini = -np.inf
        self.best_ema_epoch = 0
        self.best_ema_model_state = None
        self.patience_counter = 0

    def fit(self, train_loader, val_loader):
        print ("🔃🔃🔃Begin training Tarnet🔃🔃🔃")
        print (f"📊 Early Stop Metric: {self.early_stop_metric.upper()}")
        print (f"📊 Early Stop Start Epoch: {self.early_stop_start_epoch + 1}")
        
        if self.early_stop_metric == 'ema_qini':
            print (f"📊 Strategy: Best EMA Qini (alpha={self.ema_alpha})")
            print (f"   Restore to epoch with highest smoothed (EMA) Qini score")
            print (f"   Patience: {self.patience} epochs")
        elif self.early_stop_metric == 'qini' and self.use_ema:
            print (f"📊 Strategy: Two-Stage EMA Filter (alpha={self.ema_alpha})")
            print (f"   EMA filters noise spikes, Raw Qini determines peak height")
            print (f"   Select checkpoint: raw_qini is highest AND raw_qini >= ema_qini")
        elif self.early_stop_metric == 'qini':
            print (f"📊 Strategy: Train for {self.epoch} epochs, select model with best raw Qini score")
        elif self.early_stop_metric == 'loss':
            print (f"📊 Strategy: Train for {self.epoch} epochs, select model with lowest validation loss")
            print (f"   Patience: {self.patience} epochs")
        # TRAINING LOOP
        for epoch in range(self.epoch):
            self.model.train()
            epoch_loss=0
            for x_batch , t_batch ,y_batch in train_loader:
                    x_batch = x_batch.to(self.device)
                    
                    t_batch =t_batch.to(self.device) 
                    y_batch = y_batch.to(self.device)
                    
                    t_mask = (t_batch.squeeze(1) == 1)
                    c_mask = (t_batch.squeeze(1) == 0)
                    self.optim.zero_grad()
                    
                    #FORWARD PASS
                    y0_pred, y1_pred = self.model(x_batch)
                    
                    y_t = y_batch[t_mask]
                    y_c = y_batch[c_mask]

                    y0_pred_c = y0_pred[c_mask]
                    y1_pred_t = y1_pred[t_mask]

                    loss = outcome_loss(y_t= y_t, y_c= y_c, y1_pred=y1_pred_t, y0_pred= y0_pred_c)
                        
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optim.step()
                    epoch_loss += loss.item()
            
            # CALCULATE QINI AND LOSS
            val_qini = self.validate_qini(val_loader)
            val_loss = self.validate(val_loader, epoch)
            
            # Step the scheduler based on the selected early stopping metric
            # if self.early_stop_metric == "loss":
            #     self.scheduler.step(val_loss)
            # else: 
            #     self.scheduler.step(val_qini)
            
            # current_lr = self.optim.param_groups[0]['lr']

            # Early stopping based on selected metric
            
            # EMA QINI EARLY STOP
            if self.early_stop_metric == 'ema_qini':
                # Update EMA
                if self.ema_qini is None:
                    self.ema_qini = val_qini
                else:
                    self.ema_qini = self.ema_alpha * val_qini + (1 - self.ema_alpha) * self.ema_qini
                
                # Track best EMA Qini (always track, patience only after early_stop_start_epoch)
                if self.ema_qini > self.best_ema_qini:
                    self.best_ema_qini = self.ema_qini
                    self.best_ema_epoch = epoch
                    self.best_ema_model_state = copy.deepcopy(self.model.state_dict())
                    self.best_qini = val_qini  # Track raw qini at this epoch too
                    self.patience_counter = 0
                    best_marker = "⭐ NEW BEST EMA"
                else:
                    if epoch >= self.early_stop_start_epoch:
                        self.patience_counter += 1
                    best_marker = f"(patience: {self.patience_counter}/{self.patience})"
                
                if (epoch+1) % 1 == 0:
                    print(
                        f"Epoch {epoch+1}/{self.epoch} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Qini: {val_qini:.4f} | "
                        f"EMA Qini: {self.ema_qini:.4f} | "
                        f"Best EMA: {self.best_ema_qini:.4f} {best_marker}"
                    )
                
                # Early stopping based on patience
                if epoch >= self.early_stop_start_epoch and self.patience_counter >= self.patience:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}!")
                    print(f"   No improvement in EMA Qini for {self.patience} epochs")
                    break
            
            # LOSS EARLY STOP
            elif self.early_stop_metric == 'loss':
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_qini = val_qini  # Track qini too for reporting
                    self.best_epoch = epoch
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    self.patience_counter = 0
                    best_marker = "⭐ NEW BEST (lowest loss)"
                else:
                    self.patience_counter += 1
                    best_marker = f"(patience: {self.patience_counter}/{self.patience})"
                
                if (epoch+1) % 1 == 0:
                    print(
                        f"Epoch {epoch+1}/{self.epoch} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Total Loss: {loss.item():.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Qini: {val_qini:.4f} {best_marker}"
                    )
                
                # ONLY USE EARLYSTOP AFTER N EPOCHS
                if epoch >= self.early_stop_start_epoch and self.patience_counter >= self.patience:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}!")
                    print(f"   No improvement in validation loss for {self.patience} epochs")
                    break
              
            # QINI EARLYSTOP    
            elif self.early_stop_metric == 'qini':
                if self.use_ema:
                    # Two-Stage Strategy: EMA as noise filter, raw Qini determines peak
                    
                    # Step 1: Update EMA trend (for filtering, not for selection)
                    if self.ema_qini is None:
                        self.ema_qini = val_qini
                    else:
                        self.ema_qini = self.ema_alpha * val_qini + (1 - self.ema_alpha) * self.ema_qini
                    
                    # Step 2: Select checkpoint only if:
                    #   - raw_qini is highest so far
                    #   - AND raw_qini >= ema_qini (filters noise spikes below trend)
                    is_above_trend = val_qini >= self.ema_qini
                    is_new_peak = val_qini > self.best_qini
                    
                    if is_new_peak and is_above_trend:
                        self.best_qini = val_qini
                        self.best_epoch = epoch
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                        self.patience_counter = 0
                        best_marker = "⭐ NEW BEST (peak ≥ trend)"
                    elif is_new_peak and not is_above_trend:
                        self.patience_counter += 1
                        best_marker = f"❌ peak below trend (patience: {self.patience_counter}/{self.patience})"
                    elif not is_new_peak and is_above_trend:
                        self.patience_counter += 1
                        best_marker = f"✓ above trend but not peak (patience: {self.patience_counter}/{self.patience})"
                    else:
                        self.patience_counter += 1
                        best_marker = f"(patience: {self.patience_counter}/{self.patience})"
                    
                    if (epoch+1) % 1 == 0:
                        print(
                            f"Epoch {epoch+1}/{self.epoch} | "
                            f"Loss: {loss.item():.4f} | "
                            f"Val Loss: {val_loss:.4f} | "
                            f"Val Qini: {val_qini:.4f} {best_marker}"
                            f"EMA Trend: {self.ema_qini:.4f} | "
                            f"{best_marker}"
                        )
                    
                    # Early stopping based on patience (only after early_stop_start_epoch)
                    if epoch >= self.early_stop_start_epoch and self.patience_counter >= self.patience:
                        print(f"\n🛑 Early stopping triggered at epoch {epoch+1}!")
                        print(f"   No valid peak (raw ≥ trend) found in last {self.patience} epochs")
                        break
                else:
                    # Original: track raw Qini
                    if val_qini > self.best_qini:
                        self.best_qini = val_qini
                        self.best_epoch = epoch
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                        best_marker = "⭐ NEW BEST"
                    else:
                        best_marker = ""
                        
                    if (epoch+1) % 1 == 0:
                        print(
                            f"Epoch {epoch+1}/{self.epoch} | "
                            f"Loss: {loss.item():.4f} | "
                            f"Val Loss: {val_loss:.4f} | "
                            f"Val Qini: {val_qini:.4f} {best_marker}"
                        )
        
        # RESTORE BEST MODEL
        if self.early_stop_metric == 'ema_qini' and self.best_ema_model_state is not None:
            self.model.load_state_dict(self.best_ema_model_state)
            print(f"\n✅ Training completed! Restored model to epoch {self.best_ema_epoch+1}")
            print(f"   Best EMA Qini: {self.best_ema_qini:.4f}")
            print(f"   Raw Qini at best EMA epoch: {self.best_qini:.4f}")
            print(f"   Strategy: Selected epoch with highest smoothed (EMA) Qini")
        elif self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.early_stop_metric == 'loss':
                print(f"\n✅ Training completed! Restored model to epoch {self.best_epoch+1}")
                print(f"   Best Val Loss: {self.best_loss:.4f}")
                print(f"   Qini at best epoch: {self.best_qini:.4f}")
            elif self.early_stop_metric == 'qini' and self.use_ema:
                print(f"\n✅ Training completed! Restored model to epoch {self.best_epoch+1}")
                print(f"   Best Raw Qini: {self.best_qini:.4f}")
                print(f"   Final EMA Trend: {self.ema_qini:.4f}")
                print(f"   Strategy: Selected highest peak that stayed above EMA trend")
            else:
                print(f"\n✅ Training completed! Restored model to epoch {self.best_epoch+1} with best Qini score: {self.best_qini:.4f}")
        else:
            print(f"\n⚠️ No valid model state saved. Using final epoch model.")
            if self.early_stop_metric == 'ema_qini':
                print(f"   Final EMA Qini: {self.ema_qini:.4f}" if self.ema_qini is not None else "   EMA not initialized")
            
    def validate(self, val_loader, epoch):
        self.model.eval()
        val_loss=0
        with torch.no_grad():
            for x, t, y in val_loader:
                x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
                t_mask = (t.squeeze(1) == 1)
                c_mask = (t.squeeze(1) == 0)
                
                y0, y1 = self.model(x)
                y_t = y[t_mask]
                y_c = y[c_mask]

                y0_pred_c = y0[c_mask]
                y1_pred_t = y1[t_mask]

                loss = outcome_loss(y_t= y_t, y_c= y_c, y1_pred=y1_pred_t, y0_pred= y0_pred_c)
                
                val_loss += loss.item()
        return val_loss / len(val_loader)
    
    def validate_qini(self, val_loader):
        """Tính Qini coefficient trên validation set - GPU accelerated"""
        self.model.eval()
        y_list = []
        t_list = []
        uplift_list = []
        
        with torch.no_grad():
            for x, t, y in val_loader:
                x = x.to(self.device)
                y0_pred, y1_pred = self.model(x)
                
                # Keep tensors on GPU
                uplift = y1_pred - y0_pred
                
                y_list.append(y.to(self.device))
                t_list.append(t.to(self.device))
                uplift_list.append(uplift)
        
        # Concatenate all batches on GPU
        y_all = torch.cat(y_list, dim=0)
        t_all = torch.cat(t_list, dim=0)
        uplift_all = torch.cat(uplift_list, dim=0)

        # auqc uses NumPy/Pandas internally, so tensors must be on CPU first.
        y_all = y_all.detach().cpu().numpy()
        t_all = t_all.detach().cpu().numpy()
        uplift_all = uplift_all.detach().cpu().numpy()

        qini_score = auqc(
            y_true=y_all,
            t_true=t_all,
            uplift_pred=uplift_all,
            bins=100,
            plot=False
        )
        
        return qini_score
        
    def predict(self, x):
        self.model.eval()
        if isinstance(x, torch.Tensor):
            x = x.to(self.device, dtype=torch.float32)
        else:
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y0_pred, y1_pred = self.model(x)
        return y0_pred, y1_pred  