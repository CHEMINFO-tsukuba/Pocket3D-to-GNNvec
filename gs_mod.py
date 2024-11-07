from script_.graphsite import *
from model import *
import sys
import torch

#mol2ファイルを読み込んで、グラフデータに変換する関数
def gs(mol_path, profile_path, pop_path):
    
    node_feature, edge_index, edge_attr = graph_representation(mol_path,profile_path,pop_path)

    node_feature_tensor = torch.tensor(node_feature, dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)

    print(f"node_features: {node_feature_tensor.size()}")  # デバッグプリント
    print(f"edge_index: {edge_index_tensor.size()}")  # デバッグプリント
    print(f"edge_attr: {edge_attr_tensor.size()}")  # デバッグプリント


    return node_feature_tensor, edge_index_tensor, edge_attr_tensor



from torch_geometric.data import Data, Batch


# モデルの初期化とメッセージパッシング
def apply_message_passing(node_feature, edge_index, edge_attr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # GraphsiteClassifierモデルの初期化 (引数は必要に応じて変更)
    model = GraphsiteClassifier(
        num_classes=11, num_features=11, dim=96, train_eps=True,
        num_edge_attr=1, which_model='jknwm', num_layers=6, num_channels=3, deg=None
    ).to(device)
    
    # Dataオブジェクトを作成
    data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr)
    
    # Batchオブジェクトを作成
    batch = Batch.from_data_list([data]).to(device)
    
    # モデルにバッチデータを入力
    output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    
    return output

# Set2Setを使って固定長ベクトルに変換する関数
def vec_converter(node_feature, edge_index, edge_attr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # メッセージパッシングで更新されたノード情報を取得
    updated_node_feature = apply_message_passing(node_feature, edge_index, edge_attr)
    print("メッセージパッシング後の情報",updated_node_feature)
   
    #data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr)
    data = Data(x=updated_node_feature, edge_index=edge_index, edge_attr=edge_attr)

    batch = Batch.from_data_list([data]).to(device)
    # Set2Set アグリゲーションの実行
    set2set = Set2Set(in_channels=updated_node_feature.size(1), processing_steps=3).to(device)  # 例として3ステップ
    vec = set2set(batch.x, batch.batch)
    
    return vec

# メイン処理
if __name__ == "__main__":
    args = sys.argv
    
    # mol2ファイルのパス
    mol_path = args[1]
    profile_path = args[2]
    pop_path = args[3]

    
    # 3D構造をグラフ表現に変換
    node_feature, edge_index, edge_attr = gs(mol_path,profile_path, pop_path)
    
    # 固定長ベクトルに変換
    vec = vec_converter(node_feature, edge_index, edge_attr)
    print("固定長ベクトル:", vec)





