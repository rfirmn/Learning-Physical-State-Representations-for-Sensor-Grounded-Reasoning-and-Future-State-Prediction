"""Offline SMPL-X mesh/regressor export, or explicitly audited equivalent direct joints."""
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from eksperimen_model.datasets.m4human_dataset import CONTRACT_VERSION, decode_msgpack, read_json, read_manifest, validate_frames
from eksperimen_model.utils.m4human_runtime import atomic_json, file_sha256


def rotation(value):
    r = np.asarray(value,dtype=np.float64)
    if r.shape != (3,3) or not np.isfinite(r).all() or not np.allclose(r.T @ r,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1.,atol=1e-5):
        raise ValueError('calibration rotation not finite SO(3)')
    return r


def unit_scale(unit):
    if unit not in ('meter','millimeter'):
        raise ValueError('unit must be explicitly meter or millimeter')
    return 1. if unit == 'meter' else .001


def transform_to_radar(points_m, calibration, coordinate):
    if coordinate['source_frame'] == 'radar':
        return np.array(points_m,dtype=np.float64,copy=True)
    if coordinate['source_frame'] != 'vicon':
        raise ValueError('source_frame must be audited radar or vicon')
    r1 = rotation(calibration['vicon_to_cam_rotmatrix'])
    r2 = rotation(calibration['radar_to_cam_rotmatrix'])
    t1 = np.asarray(calibration['vicon_to_cam_tvec'],dtype=np.float64).reshape(3)*unit_scale(coordinate['vicon_to_cam_translation_unit'])
    t2 = np.asarray(calibration['radar_to_cam_tvec'],dtype=np.float64).reshape(3)*unit_scale(coordinate['radar_to_cam_translation_unit'])
    if not np.isfinite(t1).all() or not np.isfinite(t2).all():
        raise ValueError('nonfinite calibration translation')
    radar = ((points_m @ r1.T + t1)-t2) @ r2
    restored = ((radar @ r2.T+t2)-t1) @ r1
    if not np.allclose(restored,points_m,atol=1e-8,rtol=1e-8):
        raise ValueError('coordinate round trip failed')
    return radar


def target_from_joints(global_m,pelvis_index):
    global_m = np.asarray(global_m,dtype=np.float32)
    if global_m.shape != (22,3):
        raise ValueError('joint regressor/map must output22x3')
    valid_joint = np.isfinite(global_m).all(-1)
    root = global_m[pelvis_index].copy()
    root_valid = bool(valid_joint[pelvis_index])
    relative = global_m-root if root_valid else np.full((22,3),np.nan,dtype=np.float32)
    if root_valid:
        relative[pelvis_index] = 0.
    return dict(joint_global_m=global_m,root_m=root,joint_relative_m=relative,
                annotation_joint_valid=valid_joint,annotation_root_valid=np.bool_(root_valid))


class TargetExporter:
    def __init__(self, root, config, joint_map, coordinate):
        if config.get('audited') is not True or coordinate.get('audited') is not True or joint_map.get('audited') is not True:
            raise ValueError('target/unit/body map audits required')
        if coordinate.get('output_unit') != 'meter' or not coordinate.get('evidence'):
            raise ValueError('target frame evidence required')
        self.config,self.joint_map,self.coordinate = config,joint_map,coordinate
        rows = joint_map.get('source_row_indices',[])
        names = joint_map.get('ordered_names',[])
        pelvis = joint_map.get('pelvis_index',-1)
        if len(rows) != 22 or len(set(rows)) != 22 or any(type(x) is not int or x<0 for x in rows) or len(names)!=22 or not 0<=pelvis<22 or names[pelvis]!='pelvis':
            raise ValueError('audited distinct22 body rows and pelvis required')
        if config.get('method') not in ('smplx_mesh_regressor','direct_joints_equivalent'):
            raise ValueError('unknown target method')
        if config['method']=='direct_joints_equivalent' and not config.get('equivalence_evidence'):
            raise ValueError('direct joints must be audited equivalent to mesh/regressor reference')
        if config.get('immutable_snapshot') is not True:
            raise ValueError('source immutability gate required')
        import lmdb
        self.envs = {}
        for name in ('params','calib'):
            path = Path(root)/(name+'.lmdb')
            if not path.exists():
                raise FileNotFoundError(path)
            self.envs[name] = lmdb.open(str(path),subdir=path.is_dir(),readonly=True,create=False,lock=False,writemap=False,readahead=False)
        self.models = {}
        if config['method']=='smplx_mesh_regressor':
            import smplx
            if not config.get('gender_mapping') or not config.get('default_hand_face_audit') or config.get('model_output_unit') != 'meter':
                raise ValueError('SMPL-X gender/default hand-face/output-unit audit required')
            for gender, spec in config.get('assets',{}).items():
                if gender not in ('male','female','neutral') or file_sha256(spec['path']) != spec['sha256'] or spec.get('trusted_distribution') is not True:
                    raise ValueError('official trusted asset path/hash required')
                self.models[gender] = smplx.create(spec['path'],model_type='smplx',gender=gender,
                                                 use_pca=False,num_betas=config['num_betas'],ext=Path(spec['path']).suffix.lstrip('.'))
                self.models[gender].eval().requires_grad_(False)
            if not self.models:
                raise ValueError('SMPL-X assets not configured; no download/default gender')

    def read(self,name,key):
        with self.envs[name].begin(write=False) as tx:
            raw = tx.get(key.encode('utf-8'))
        return decode_msgpack(raw)

    def export(self,row):
        param = self.read('params',row['source_key'])
        calib = self.read('calib',row.get('calibration_key',row['source_key'])) if self.coordinate['source_frame']!='radar' else {}
        indices = self.joint_map['source_row_indices']
        if self.config['method']=='direct_joints_equivalent':
            joints = np.asarray(param['joints'],dtype=np.float64)
            if joints.ndim!=2 or joints.shape[1]!=3 or max(indices)>=len(joints):
                raise ValueError('direct joints shape/topology mismatch')
            joints = joints[indices]*unit_scale(self.config['direct_joint_unit'])
            global_m = transform_to_radar(joints,calib,self.coordinate)
        else:
            for key in ('betas','pose_body','root_orient','trans','gender'):
                if key not in param:
                    raise ValueError(f'incomplete SMPL-X params:{key}')
            gender_key = str(param['gender'].item() if isinstance(param['gender'],np.ndarray) and param['gender'].ndim==0 else param['gender'])
            gender = self.config['gender_mapping'].get(gender_key)
            if gender not in self.models:
                raise ValueError('source gender lacks audited mapping/asset')
            shapes = {'betas':(self.config['num_betas'],),'pose_body':(63,),'root_orient':(3,),'trans':(3,)}
            arrays = {}
            for key,shape in shapes.items():
                arrays[key] = np.asarray(param[key],dtype=np.float32)
                if arrays[key].shape!=shape or not np.isfinite(arrays[key]).all():
                    raise ValueError(f'SMPL-X param shape/finite gate:{key}')
            arrays['trans'] *= unit_scale(self.config['parameter_translation_unit'])
            model = self.models[gender]
            kwargs = {target:torch.from_numpy(arrays[key][None]) for target,key in [('betas','betas'),('body_pose','pose_body'),('global_orient','root_orient'),('transl','trans')]}
            # Missing hand/face fields are only allowed with explicit audited defaults.
            defaults = self.config.get('default_hand_face_values',{})
            required = {'left_hand_pose':45,'right_hand_pose':45,'jaw_pose':3,'leye_pose':3,'reye_pose':3,'expression':self.config['num_expression_coeffs']}
            for name,n in required.items():
                value = np.asarray(defaults.get(name),dtype=np.float32)
                if value.shape!=(n,) or not np.isfinite(value).all():
                    raise ValueError(f'explicit audited hand/face default required:{name}')
                kwargs[name] = torch.from_numpy(value[None])
            with torch.no_grad():
                vertices = model(**kwargs).vertices[0].double().cpu().numpy()
            radar_vertices = transform_to_radar(vertices,calib,self.coordinate)
            regressor = model.J_regressor.detach().double().cpu().numpy()[indices]
            if regressor.shape[1]!=len(vertices) or not np.allclose(regressor.sum(1),1.,atol=1e-5):
                raise ValueError('audited joint regressor must be affine-normalized')
            global_m = regressor @ radar_vertices
        return target_from_joints(global_m,self.joint_map['pelvis_index'])

    def close(self):
        for env in self.envs.values():
            env.close()


def export_targets(root,manifest,output,config_path,joint_map_path,coordinate_path):
    rows = validate_frames(read_manifest(manifest))
    output = Path(output)
    if output.resolve()==Path(root).resolve() or Path(root).resolve() in output.resolve().parents:
        raise ValueError('target output must be outside source')
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name+'.incomplete')
    temporary.mkdir(parents=True,exist_ok=False)
    config,joint_map,coordinate = map(read_json,(config_path,joint_map_path,coordinate_path))
    engine = TargetExporter(root,config,joint_map,coordinate)
    fields = {'joint_global_m':((len(rows),22,3),np.float32),'root_m':((len(rows),3),np.float32),'joint_relative_m':((len(rows),22,3),np.float32),'annotation_joint_valid':((len(rows),22),bool),'annotation_root_valid':((len(rows),),bool)}
    arrays = {name:np.lib.format.open_memmap(temporary/(name+'.npy'),mode='w+',dtype=dtype,shape=shape) for name,(shape,dtype) in fields.items()}
    errors = 0
    try:
        with (temporary/'manifest.jsonl').open('w',encoding='utf-8') as out, (temporary/'rejections.jsonl').open('w',encoding='utf-8') as rejected:
            for i,row in enumerate(rows):
                try:
                    target = engine.export(row)
                except (ValueError,KeyError) as exc:
                    errors += 1
                    target = target_from_joints(np.full((22,3),np.nan,dtype=np.float32),joint_map['pelvis_index'])
                    rejected.write(json.dumps({'frame_uid':row['frame_uid'],'phase':'target_export','reason':str(exc)})+'\n')
                for name,arr in arrays.items():
                    arr[i] = target[name]
                out.write(json.dumps(row,allow_nan=False)+'\n')
        for arr in arrays.values():
            arr.flush()
        metadata = {'contract_version':CONTRACT_VERSION,'complete':True,'data_kind':config.get('data_kind','m4human'),'count':len(rows),'invalid_target_frames':errors,
                    'lineage':{name:file_sha256(path) for name,path in [('frame_manifest_hash',manifest),('joint_map_hash',joint_map_path),('coordinate_hash',coordinate_path),('export_recipe_hash',config_path)]},
                    'method':config['method'],'pelvis_index':joint_map['pelvis_index'], 'arrays':{name:file_sha256(temporary/(name+'.npy')) for name in fields},
                    'manifest_hash':file_sha256(temporary/'manifest.jsonl')}
        atomic_json(temporary/'metadata.json',metadata)
        for arr in arrays.values():
            if hasattr(arr, '_mmap') and arr._mmap is not None:
                arr._mmap.close()
        del arrays
        import gc
        gc.collect()
        temporary.rename(output)
    finally:
        engine.close()
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='datasets/m4human')
    for name in ('manifest','output-dir','export-config','joint-map','coordinate-audit'):
        p.add_argument('--'+name,required=True)
    a = p.parse_args()
    print(json.dumps(export_targets(a.root,a.manifest,a.output_dir,a.export_config,a.joint_map,a.coordinate_audit),indent=2))

if __name__=='__main__':
    main()
