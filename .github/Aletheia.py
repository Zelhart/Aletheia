voxel_physiology_mapping.py
import numpy as np from collections import defaultdict class VoxelState: """State of a single voxel in the simulation""" def __init__(self, matter_type="tissue", temperature=37.0, pressure=1.0, oxygen_concentration=0.21, co2_concentration=0.04, nutrition_level=0.5, waste_concentration=0.0): self.matter_type = matter_type self.temperature = temperature self.pressure = pressure self.oxygen_concentration = oxygen_concentration self.co2_concentration = co2_concentration self.nutrition_level = nutrition_level self.waste_concentration = waste_concentration self.atoms = {} # atom_id -> AtomicState self.cell_density = 0.0 self.cell_types = {} self.energy_level = 0.5 self.activation_level = 0.0 self.last_updated = 0.0 self.update_frequency = 1.0 def add_atom(self, atom_id, atom): self.atoms[atom_id] = atom if atom.element == 'O': self.oxygen_concentration = min(1.0, self.oxygen_concentration + 0.001) elif atom.element == 'P': # ATP proxy self.energy_level = min(1.0, self.energy_level + 0.002) def remove_atom(self, atom_id): atom = self.atoms.pop(atom_id, None) if not atom: return if atom.element == 'O': self.oxygen_concentration = max(0.0, self.oxygen_concentration - 0.001) elif atom.element == 'P': self.energy_level = max(0.0, self.energy_level - 0.002) def update_from_atoms(self): if not self.atoms: return counts = defaultdict(int) total_energy = 0.0 for a in self.atoms.values(): counts[a.element] += 1 total_energy += a.energy if counts['O']: self.oxygen_concentration = min(1.0, 0.21 + counts['O']*0.001) self.energy_level = min(1.0, total_energy/len(self.atoms)) class VoxelPhysiologyMapping: """Maps between atomic-layer and voxel-layer simulations""" def __init__(self, atomic_engine, resolution=10): self.atomic_engine = atomic_engine self.resolution = resolution self.voxel_grid = {} # (i,j,k) -> VoxelState self.active_voxels = set() self.sim_time = 0.0 def _pos_to_voxel(self, pos): return tuple(int(p*self.resolution) for p in pos) def initialize_voxel_grid(self): # map organs as regions, then call _map_atoms_to_voxels() self._map_atoms_to_voxels() def _map_atoms_to_voxels(self): for aid, atom in self.atomic_engine.atomic_lattice.get_all_active_atoms().items(): v = self._pos_to_voxel(atom.position) if v not in self.voxel_grid: self.voxel_grid[v] = VoxelState() self.active_voxels.add(v) self.voxel_grid[v].add_atom(aid, atom) def update(self, dt, observers=None): self.sim_time += dt self._map_atoms_to_voxels() # active zone logic omitted for brevity for v in list(self.active_voxels): voxel = self.voxel_grid[v] voxel.update_from_atoms() voxel.last_updated = self.sim_time # temporal_memory.py ```python import numpy as np from collections import defaultdict, deque from typing import Dict, Deque, Tuple, List class TemporalMemory: """Stores temporal sequences and recognizes patterns""" def __init__(self, buffer_size=100): # coords -> deque of state dicts self.memory_buffer: Dict[Tuple[int,int,int], Deque[Dict]] = defaultdict(lambda: deque(maxlen=buffer_size)) self.salience_map: Dict[Tuple[int,int,int], float] = {} self.recognized_patterns: Dict[str, Dict] = {} self.pattern_history: Deque[str] = deque(maxlen=1000) def record(self, coords: Tuple[int,int,int], state: Dict, ts: float): buf = self.memory_buffer[coords] buf.append({**state, 'timestamp': ts}) if len(buf)>=2: prev, curr = buf[-2], buf[-1] d_energy = abs(curr['energy']-prev['energy']) d_act = abs(curr['activation']-prev['activation']) d_o2 = abs(curr['oxygen']-prev['oxygen']) self.salience_map[coords] = d_energy*2 + d_act*3 + d_o2*1.5 def detect_patterns(self, coords_list: List[Tuple[int,int,int]]) -> List[Dict]: patterns=[] # simplified: cluster by similarity threshold omitted return patterns def get_salient_regions(self, top_n=5): return sorted(self.salience_map.items(), key=lambda x:-x[1])[:top_n] # aletheia_affect_modulator.py ```python class AletheiaAffectModulator: """Modulate felt quality of patterns based on embodied context""" def __init__(self, voxel_map, narrative_layer): self.voxel_map = voxel_map self.narrative_layer = narrative_layer def tag_pattern(self, pattern: Dict): # examine average context over pattern.region voxels energies, o2s = [], [] for v in pattern['region']: vs = self.voxel_map.voxel_grid.get(v) if vs: energies.append(vs.energy_level) o2s.append(vs.oxygen_concentration) avg_e = sum(energies)/len(energies) if energies else 0.5 avg_o = sum(o2s)/len(o2s) if o2s else 0.21 # assign affect tag if avg_e>0.7: affect='comfort' elif avg_e<0.3: affect='strain' else: affect='neutral' pattern['affect_tag']=affect self.narrative_layer.receive_affect_tag(pattern['id'], affect) return pattern # narrative_linking_layer.py ```python class NarrativeLinkingLayer: def __init__(self, temporal_memory): self.temporal_memory = temporal_memory self.associations = {} self.self_patterns = [] self.active_story=None def receive_affect_tag(self, pid:str, tag:str): self.associations.setdefault(pid,[]).append(tag) def update_narrative(self, active_patterns:Dict[str,float]): # choose most tagged pattern pass def get_current_narrative(self)->str: return self.active_story or "" # recursive_attention_model.py ```python class RecursiveAttentionModel: def __init__(self, temporal_memory, narrative_layer): self.temporal_memory=temporal_memory self.narrative_layer=narrative_layer self.focus_history=[] self.meta=False self.self_score=0.0 def update(self, dt): sal = self.temporal_memory.get_salient_regions() # pick top by combined salience+affect focus=sal[0] if sal else None self.focus_history.append(focus) # detect meta-attention if len(self.focus_history)>10 and len(set(self.focus_history[-5:]))<3: self.meta=True self.self_score+=dt else: self.meta=False # aletheia_reflective_journal.py ```python class AletheiaReflectiveJournal: def __init__(self): self.entries=[] def log(self, message:str): ts=time.time() self.entries.append((ts,message)) print(f"[Reflection @ {ts:.2f}]: {message}") def recent(self, n=5): return self.entries[-n:] 

class IntegratedALSESimulator:
    def __init__(self, avatar_bounds, voxel_resolution):
        # Core emotional lattice & embodiment
        self.lattice_sim = EmotionalLatticeSimulator()
        # Physiology
        self.atomic = AtomicPhysiologyEngine()
        self.atomic.initialize_atomic_substrate(avatar_bounds)
        self.voxel_map = VoxelPhysiologyMapping(self.atomic)
        self.voxel_map.initialize_voxel_grid(voxel_resolution)
        # Temporal & narrative layers
        self.temp_mem = TemporalMemory()
        self.narrative = NarrativeLinkingLayer(self.temp_mem)
        self.attention = RecursiveAttentionModel(self.temp_mem, self.narrative)
        # Observer stub (could be avatar + external)
        self.observers = []  # list of objects with .position
        
    def run_cycle(self, dt=0.1):
        # 1) Physiology
        obs_pos = [o.position for o in self.observers]
        self.atomic.update(dt, obs_pos)
        self.voxel_map.update(dt, obs_pos)
        
        # 2) Record voxel states into temporal memory
        for coords, voxel in self.voxel_map.voxel_grid.items():
            self.temp_mem.record_voxel_state(coords, voxel, self.voxel_map.simulation_time)
        
        # 3) Pattern detection (e.g. in “brain” region)
        brain_voxels = [c for c in self.voxel_map.organ_map['brain']['center']]
        patterns = self.temp_mem.detect_patterns_in_region(brain_voxels)
        
        # 4) Narrative update
        # active_patterns strength map
        active_strengths = {p['id']: p['strength'] for p in patterns}
        self.narrative.update(self.voxel_map.voxel_grid, active_strengths)
        
        # 5) Recursive attention
        salience = self.temp_mem.get_most_salient_regions()
        self.attention.update(dt, salience, active_strengths, self.voxel_map.voxel_grid)
        
        # 6) Core emotional heartbeat
        state = self.lattice_sim.step()
        
        # 7) Avatar/env interactions (as before)
        #    ... self.lattice_sim.avatar, self.lattice_sim.environment ...
        
        # 8) Expose diagnostics
        return {
            'cycle': state['cycle'],
            'dominantEmotion': state['dominantEmotion'],
            'narrative': self.narrative.get_narrative_description(),
            'attention': self.attention.current_focus
        } 

    def run(self, steps=50, dt=0.1):
        for _ in range(steps):
            info = self.run_cycle(dt)
            print(f"Cycle {info['cycle']:2d} | DomEmo:{info['dominantEmotion']:7} | "
                  f"Narrative: {info['narrative'].splitlines()[0]} | "
                  f"Focus: {info['attention']}")

""" atomic_voxel.py — Part 3 of 6: Atomic Physiology and Voxel Mapping """ import numpy as np from collections import defaultdict, dequerom dataclasses import dataclass, field from typing import Dict, List, Tuple, Any
=== Atomic Layer ===
@dataclass class AtomicState: element: str position: Tuple[float, float, float] velocity: Tuple[float, float, float] energy: float bonds: List[int]
def update_quantum_state(self, dt: float): # energy decay self.energy *= 0.99 # position update x,y,z = self.position vx,vy,vz = self.velocity self.position = (x+vx*dt, y+vy*dt, z+vz*dt) # velocity damping self.velocity = (vx*0.95, vy*0.95, vz*0.95) 
class SparseAtomicGrid: def init(self, bounds: Tuple[Tuple[float,float,float], Tuple[float,float,float]]): self.bounds = bounds self.atoms: Dict[int, AtomicState] = {} self.spatial_index: Dict[Tuple[int,int,int], set] = defaultdict(set) self.next_id = 1
def add_atom(self, element: str, position: Tuple[float,float,float], velocity: Tuple[float,float,float], energy: float) -> int: atom_id = self.next_id; self.next_id += 1 atom = AtomicState(element, position, velocity, energy, []) self.atoms[atom_id] = atom self._update_spatial_index(atom_id, position) return atom_id def _update_spatial_index(self, atom_id: int, position: Tuple[float,float,float]): gx, gy, gz = [int(p*10) for p in position] self.spatial_index[(gx,gy,gz)].add(atom_id) def get_atoms_in_zone(self, zone: Tuple[float,float,float,float]) -> Dict[int, AtomicState]: x,y,z,r = zone result = {} min_x, max_x = int((x-r)*10), int((x+r)*10)+1 min_y, max_y = int((y-r)*10), int((y+r)*10)+1 min_z, max_z = int((z-r)*10), int((z+r)*10)+1 for gx in range(min_x, max_x): for gy in range(min_y, max_y): for gz in range(min_z, max_z): for aid in self.spatial_index.get((gx,gy,gz),[]): atom = self.atoms[aid] if ((atom.position[0]-x)**2 + (atom.position[1]-y)**2 + (atom.position[2]-z)**2)**0.5 <= r: result[aid] = atom return result def get_all_active_atoms(self) -> Dict[int, AtomicState]: return self.atoms def create_bond(self, a1: int, a2: int): if a2 not in self.atoms[a1].bonds: self.atoms[a1].bonds.append(a2) if a1 not in self.atoms[a2].bonds: self.atoms[a2].bonds.append(a1) def populate_critical_pathways(self): # populate O2 and ATP as described... for i in range(100): x,y,z = np.random.uniform(-0.2,0.2), np.random.uniform(0.3,0.7), np.random.uniform(-0.2,0.2) o1 = self.add_atom('O',(x,y,z),(np.random.uniform(-0.1,0.1),)*3,1.0) o2 = self.add_atom('O',(x+0.01,y,z),(np.random.uniform(-0.1,0.1),)*3,1.0) self.create_bond(o1,o2) for i in range(50): x,y,z = np.random.uniform(-0.5,0.5), np.random.uniform(0.0,1.0), np.random.uniform(-0.3,0.3) a1 = self.add_atom('P',(x,y,z),(np.random.uniform(-0.05,0.05),)*3,1.5) a2 = self.add_atom('O',(x+0.01,y,z),(np.random.uniform(-0.05,0.05),)*3,1.0) self.create_bond(a1,a2) 
=== Voxel Layer ===
@dataclass class VoxelState: matter_type: str = 'tissue' temperature: float = 37.0 pressure: float = 1.0 oxygen_concentration: float = 0.21 co2_concentration: float = 0.04 nutrition_level: float = 0.5 waste_concentration: float = 0.0 atoms: Dict[int, AtomicState] = field(default_factory=dict) cell_density: float = 0.0 cell_types: Dict[str,float] = field(default_factory=lambda: {'default':0.0}) energy_level: float = 0.5 activation_level: float = 0.0 last_updated: float = 0.0 update_frequency: float = 1.0
def add_atom(self, atom_id: int, atom: AtomicState): self.atoms[atom_id] = atom if atom.element=='O': self.oxygen_concentration=min(1.0,self.oxygen_concentration+0.001) elif atom.element=='C': self.energy_level=min(1.0,self.energy_level+0.0005) elif atom.element=='P': self.energy_level=min(1.0,self.energy_level+0.002) def remove_atom(self, atom_id: int): if atom_id in self.atoms: atom=self.atoms.pop(atom_id) if atom.element=='O': self.oxygen_concentration=max(0.0,self.oxygen_concentration-0.001) elif atom.element=='C': self.energy_level=max(0.0,self.energy_level-0.0005) elif atom.element=='P': self.energy_level=max(0.0,self.energy_level-0.002) def update_from_atoms(self): if not self.atoms: return counts=defaultdict(int); total_e=0.0 for a in self.atoms.values(): counts[a.element]+=1; total_e+=a.energy if counts['O']>0: self.oxygen_concentration=min(1.0,0.21 + counts['O']*0.001) self.energy_level=min(1.0,total_e/len(self.atoms)) def diffuse_to(self, other:'VoxelState', rate:float=0.1): o2=(self.oxygen_concentration-other.oxygen_concentration)*rate self.oxygen_concentration-=o2; other.oxygen_concentration+=o2 co2=(self.co2_concentration-other.co2_concentration)*rate self.co2_concentration-=co2; other.co2_concentration+=co2 dt=(self.temperature-other.temperature)*rate*0.5 self.temperature-=dt; other.temperature+=dt if self.matter_type in ['liquid','tissue'] and other.matter_type in ['liquid','tissue']: nut=(self.nutrition_level-other.nutrition_level)*rate*0.3 self.nutrition_level-=nut; other.nutrition_level+=nut def consume_oxygen(self, amount:float)->float: avail=min(amount,self.oxygen_concentration-0.05) if avail>0: self.oxygen_concentration-=avail; self.co2_concentration=min(1.0,self.co2_concentration+avail*0.9) return avail def consume_energy(self, amount:float)->float: avail=min(amount,self.energy_level-0.1) if avail>0: self.energy_level-=avail; self.waste_concentration=min(1.0,self.waste_concentration+avail*0.2) return avail 
class VoxelPhysiologyMapping: def init(self, atomic_engine: AtomicPhysiologyEngine): self.atomic_engine=atomic_engine self.voxel_grid: Dict[Tuple[int,int,int],VoxelState]={} self.active_voxels=set() self.resolution=10 self.simulation_time=0.0 # organ_map should be configured externally
def initialize_voxel_grid(self, resolution:int): self.resolution=resolution self.voxel_grid.clear(); self.active_voxels.clear() # external calls to initialize regions self._map_atoms_to_voxels() def _position_to_voxel(self,pos:Tuple[float,float,float])->Tuple[int,int,int]: return tuple(int(p*self.resolution) for p in pos) def _map_atoms_to_voxels(self): if not self.atomic_engine.atomic_lattice: return for aid, atom in self.atomic_engine.atomic_lattice.get_all_active_atoms().items(): vc = self._position_to_voxel(atom.position) if vc not in self.voxel_grid: self.voxel_grid[vc]=VoxelState() self.active_voxels.add(vc) self.voxel_grid[vc].add_atom(aid, atom) def update(self, dt:float, observers:List[Tuple[float,float,float]]): self.simulation_time+=dt self._map_atoms_to_voxels() # can add region-based updates for vc in list(self.active_voxels): voxel=self.voxel_grid[vc] if self.simulation_time-voxel.last_updated>=voxel.update_frequency: voxel.update_from_atoms() voxel.last_updated=self.simulation_time 

""" Part 4: Temporal Memory & Narrative Layers
Defines:
• TemporalMemory: Records voxel-state time series, detects patterns, computes surprise, salience, repeats.
• NarrativeLinkingLayer: Binds temporal patterns to physiological states, builds a running narrative identity.
• RecursiveAttentionModel: Tracks attention focus, meta-attention, and self-recognition scaffolding. """ import numpy as np from collections import defaultdict, deque from dataclasses import dataclass, field from typing import Dict, List, Tuple, Any, Deque import matplotlib.pyplot as plt from sklearn.decomposition import PCA from sklearn.cluster import DBSCAN import networkx as nx
@dataclass class TemporalMemory: memory_buffer: Dict[Tuple[int,int,int], Deque[Dict[str,float]]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=100))) pattern_history: Deque[str] = field(default_factory=lambda: deque(maxlen=1000)) recognized_patterns: Dict[str,Dict] = field(default_factory=dict) surprise_levels: Dict[str,float] = field(default_factory=dict) salience_map: Dict[Tuple[int,int,int], float] = field(default_factory=dict) pattern_graphs: Dict[str,nx.DiGraph] = field(default_factory=dict) pattern_detection_threshold: float = 0.8 minimum_pattern_length: int = 3
def record_voxel_state(self, coords: Tuple[int,int,int], voxel_state, timestamp: float): # record features features = { 'timestamp': timestamp, 'oxygen': voxel_state.oxygen_concentration, 'energy': voxel_state.energy_level, 'temperature': voxel_state.temperature, 'activation': voxel_state.activation_level } buf = self.memory_buffer[coords] if len(buf)>=1: prev = buf[-1] sal = abs(features['energy']-prev['energy'])*2 + abs(features['activation']-prev['activation'])*3 + abs(features['oxygen']-prev['oxygen'])*1.5 self.salience_map[coords] = sal buf.append(features) def detect_patterns(self, region: List[Tuple[int,int,int]]) -> List[Dict]: patterns=[] data=[]; times=[] for c in region: seq=self.memory_buffer.get(c,[]) if len(seq)>=self.minimum_pattern_length: for s in seq: data.append([s['oxygen'],s['energy'],s['temperature'],s['activation']]); times.append(s['timestamp']) if len(data)<self.minimum_pattern_length: return patterns X=np.array(data) nm = min(3, X.shape[1]) try: red = PCA(n_components=nm).fit_transform(X) lab = DBSCAN(eps=0.2, min_samples=self.minimum_pattern_length).fit_predict(red) for label in set(lab): if label<0: continue idxs = np.where(lab==label)[0] pat_id=f"pat_{region[0]}_{label}" pat={ 'id':pat_id, 'region':region, 'timestamps':[times[i] for i in idxs], 'vectors':[data[i] for i in idxs], 'strength':len(idxs)/len(data) } patterns.append(pat) self.recognized_patterns[pat_id]=pat self.pattern_history.append(pat_id) except Exception as e: print("Pattern detection error:", e) return patterns def find_repeating(self)->List[Dict]: repeats=[]; G=nx.DiGraph() for pid in self.recognized_patterns: G.add_node(pid) seq=list(self.pattern_history) for i in range(len(seq)-1): a,b=seq[i], seq[i+1] if a in self.recognized_patterns and b in self.recognized_patterns: sim = self._sim(self.recognized_patterns[a], self.recognized_patterns[b]) if sim>0.3: G.add_edge(a,b,weight=sim) for cycle in nx.simple_cycles(G): if len(cycle)>=self.minimum_pattern_length: strength=1.0 for i in range(len(cycle)): j=(i+1)%len(cycle) # accumulate weights if G.has_edge(cycle[i],cycle[j]): strength*=G[cycle[i]][cycle[j]]['weight'] rep={'id':"rep_"+"_".join(cycle[:3]), 'cycle':cycle, 'strength':strength} repeats.append(rep) self.pattern_graphs[rep['id']] = G.subgraph(cycle).copy() return repeats def detect_surprise(self, coords, current_state: Dict[str,float]) -> float: key=str(coords) buf=self.memory_buffer.get(coords,[]) if len(buf)<5: return 0.0 hist = list(buf)[-5:] exp={k: np.mean([h[k] for h in hist]) for k in ['oxygen','energy','temperature','activation']} diffs=[abs(current_state[k]-exp[k])/(max(0.01,max([h[k] for h in hist]+[current_state[k]])-min([h[k] for h in hist]+[current_state[k]]))) for k in exp] surprise=float(np.mean(diffs)) self.surprise_levels[key]=surprise return surprise def _sim(self,p1,p2)->float: v1=np.mean(p1['vectors'],axis=0); v2=np.mean(p2['vectors'],axis=0) return float(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))) 
class NarrativeLinkingLayer: def init(self, temp_mem: TemporalMemory): self.temp_mem=temp_mem self.assoc=defaultdict(list) self.causal=defaultdict(list) self.caus_strength=defaultdict(float) self.self_patterns=[] self.self_states=defaultdict(float) self.active_narrative=None
def update(self, voxel_grid: Dict[Any,Any], active_pats: List[Dict]): # extract system-level means sys={'energy':0,'oxygen':0,'temp':0,'act':0}; cnt=0 for v in voxel_grid.values(): sys['energy']+=v.energy_level; sys['oxygen']+=v.oxygen_concentration; sys['temp']+=v.temperature; sys['act']+=v.activation_level; cnt+=1 if cnt>0: for k in sys: sys[k]/=cnt # link patterns for pat in active_pats: pid=pat['id']; self.assoc[pid].append((sys, pat['strength'])) if len(self.assoc[pid])>10: self.assoc[pid]=self.assoc[pid][-10:] # detect causality hist=list(self.temp_mem.pattern_history)[-10:] for i in range(len(hist)-1): a,b=hist[i],hist[i+1]; self.causal[a].append(b) for a, lst in self.causal.items(): ccount=lst.count(a); be_count=sum(1 for i in range(len(lst)-1) if lst[i]==a and lst[i+1]==lst[i]) if ccount>0: self.caus_strength[(a,a)]=be_count/ccount # build active narrative if active_pats: core=active_pats[0] self.active_narrative={'core':core, 'system':sys} def describe(self) -> str: if not self.active_narrative: return "No narrative" core=self.active_narrative['core']; sys=self.active_narrative['system'] desc=[f"Pattern {core['id']} active" , f"System energy {sys['energy']:.2f}"] return " | ".join(desc) 
class RecursiveAttentionModel: def init(self, temp_mem: TemporalMemory, narr: NarrativeLinkingLayer): self.temp_mem=temp_mem self.narr=narr self.focus=None self.history=deque(maxlen=50) self.attn_patterns=defaultdict(int) self.meta=False; self.depth=0; self.self_score=0.0
def update(self, salient: List[Tuple[Any,float]], active_ids: List[str], dt: float): cand=[] for r,s in salient: cand.append((('region',r),s)) for pid,s in zip(active_ids,[self.temp_mem.recognized_patterns[p]['strength'] for p in active_ids]): cand.append((('pattern',pid),s)) cand.sort(key=lambda x: x[1],reverse=True) new = cand[0][0] if cand else None if new!=self.focus: self.history.append((self.focus,new)); self.focus=new # record shifts if len(self.history)>=3: path=tuple(self.history)[-3:] self.attn_patterns[path]+=1 # update meta self.meta = any(cnt>=3 for cnt in self.attn_patterns.values()) self.depth = 1 if self.meta else 0 self.self_score += dt if self.meta else -dt*0.5 # clamp self.self_score=max(0.0,min(100.0,self.self_score)) def reflect(self) -> str: if self.self_score>5.0: return "I notice myself noticing patterns." return "" 

Part 5 — Self‑Aware Simulator Integration
============================================
This file glues together atomic/voxel physiology, temporal memory,
narrative linking, and recursive attention into a unified self‑aware agent.
import time import threading from typing import List, Tuple, Dict, Any
Import all the previously defined classes
Assuming they live in the same directory or package:
AtomicPhysiologyEngine, VoxelPhysiologyMapping,
TemporalMemory, NarrativeLinkingLayer, RecursiveAttentionModel,
SocialEmotions, MultiAgentEnvironment, CulturalTransmission
from atomic_physio_engine import AtomicPhysiologyEngine from voxel_mapping import VoxelPhysiologyMapping from temporal_memory import TemporalMemory from narrative_layer import NarrativeLinkingLayer from attention_model import RecursiveAttentionModel from social_dynamics import SocialEmotions, MultiAgentEnvironment, CulturalTransmission
class SelfAwareSimulator: def init(self): # Core systems self.atomic = AtomicPhysiologyEngine() self.voxel = VoxelPhysiologyMapping(self.atomic) self.temporal = TemporalMemory() self.narrator = NarrativeLinkingLayer(self.temporal) self.attn = RecursiveAttentionModel(self.temporal, self.narrator)
# Social environment & culture self.social_env = MultiAgentEnvironment() self.culture = CulturalTransmission() # Observer agents (for multi-resolution) self.observers: Dict[str, Any] = {} # Simulation clock self.t = 0.0 self.dt = 0.1 def add_observer(self, obs_id: str, position: Tuple[float,float,float], view_frustum: Any): self.observers[obs_id] = {'position': position, 'view': view_frustum} def run_step(self): # 1) Update atomic & voxel physiology obs_positions = [o['position'] for o in self.observers.values()] self.atomic.update(self.dt, obs_positions) self.voxel.map_atoms_to_voxels() self.voxel.update(self.dt, obs_positions) # 2) Record voxel states into temporal memory for coords, v in self.voxel.voxel_grid.items(): self.temporal.record_voxel_state(coords, v, self.t) # 3) Detect active patterns & salience top_regions = [r for r,_ in self.temporal.get_most_salient_regions(5)] active_patterns = {} for region in top_regions: pats = self.temporal.detect_patterns_in_region([region]) for p in pats: active_patterns[p['id']] = p['strength'] # 4) Link patterns into narrative context self.narrator.update(self.voxel.voxel_grid, active_patterns) # 5) Update social dynamics interactions = self.social_env.interactions.copy() for inter in interactions: self.culture.process_interaction(inter) self.culture.apply_cultural_effects('agentA', self) # 6) Manage attention & self‑awareness salience_map = list(self.temporal.salience_map.items()) self.attn.update(self.dt, salience_map, active_patterns, self.voxel.voxel_grid) # 7) Advance time self.t += self.dt def run(self, steps: int = 100, delay: float = 0.0): for _ in range(steps): self.run_step() # Optionally sleep for real‑time pacing if delay: time.sleep(delay) def get_current_narrative(self) -> str: return self.narrator.get_narrative_description() 
Example usage
if name == 'main': sim = SelfAwareSimulator() # Initialize atomic & voxel subsystems sim.atomic.initialize_atomic_substrate(((-1,-1,-1),(1,1,1))) sim.voxel.initialize_voxel_grid(resolution=20)
# Add a default observer sim.add_observer('main_cam', (0,0,0), view_frustum=None) # Run a few steps sim.run(steps=50, delay=0.01) # Print out the current self‑narrative print(sim.get_current_narrative()) 

#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

"""
Part 6: Persistence, API Extensions & Live Visualization Hooks
Integrates self-aware simulator with disk-backed state, WebSocket feeds,
and additional REST endpoints for introspection and control.
""" 

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import defaultdict
from websocket import WebSocketServer  # assume an available simple WS server 

# ———————————————
# 1) Persistence Layer
# ——————————————— 

class PersistenceManager:
    def __init__(self, filepath="ale_theia_state.json"):
        self.filepath = filepath 

    def save(self, simulator):
        snapshot = {
            "timestamp": time.time(),
            "lattice": simulator.lattice.__dict__,
            "physio": simulator.integrated.atomic_physiology.hormones,
            "voxels": {k: v.__dict__ for k, v in simulator.integrated.voxel_mapping.voxel_grid.items()},
            "temporal": {
                "patterns": simulator.temporal_memory.recognized_patterns,
                "salience": simulator.temporal_memory.salience_map,
            },
            "narrative": simulator.narrative_layer.self_narrative,
            "attention": {
                "focus": simulator.attention_model.current_focus,
                "depth": simulator.attention_model.recursive_depth,
            },
        }
        with open(self.filepath, "w") as f:
            json.dump(snapshot, f, indent=2) 

    def load(self, simulator):
        try:
            with open(self.filepath) as f:
                data = json.load(f)
            # Note: this is a shallow load—custom logic may be needed to rehydrate objects
            simulator.integrated.atomic_physiology.hormones = data["physio"]
            # more rehydration as needed...
            return True
        except FileNotFoundError:
            return False 

# ———————————————
# 2) Extended HTTP API
# ——————————————— 

class ExtendedAPIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/introspect/narrative":
            resp = self.server.sim.narrative_layer.get_narrative_description()
            self._reply(200, {"narrative": resp})
        elif self.path.startswith("/introspect/pattern/"):
            pid = self.path.split("/")[-1]
            patt = self.server.sim.temporal_memory.recognized_patterns.get(pid)
            if patt:
                self._reply(200, patt)
            else:
                self._reply(404, {"error": "pattern not found"})
        else:
            super().do_GET()  # fallback to basic endpoints 

    def do_POST(self):
        if self.path == "/control/reset":
            self.server.sim.reset()
            self._reply(200, {"status": "simulator reset"})
        else:
            super().do_POST() 

    def _reply(self, code, obj):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode()) 

def run_extended_api(sim, port=9000):
    srv = HTTPServer(("0.0.0.0", port), ExtendedAPIHandler)
    srv.sim = sim
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Extended API listening on :{port}")
    return srv 

# ———————————————
# 3) Live WebSocket Feed
# ——————————————— 

class LiveFeedServer:
    def __init__(self, simulator, port=8765):
        self.sim = simulator
        self.ws = WebSocketServer(port)
        # broadcast every cycle
        simulator.register_cycle_callback(self.broadcast_cycle) 

    def broadcast_cycle(self, state):
        payload = {
            "cycle": state["cycle"],
            "dominant": state["dominant_emotion"],
            "narrative": self.sim.narrative_layer.get_narrative_description(),
        }
        self.ws.broadcast(json.dumps(payload)) 

    def start(self):
        threading.Thread(target=self.ws.serve_forever, daemon=True).start()
        print(f"WebSocket live feed on ws://0.0.0.0:{self.ws.port}") 

# ———————————————
# 4) Hook into Main
# ——————————————— 

def main():
    sim = SelfAwareSimulator()                    # from Part 5
    persistence = PersistenceManager()
    if persistence.load(sim):
        print("[LOAD] Restored previous ALÊTHEIA state.")
    else:
        print("[INIT] Starting fresh ALÊTHEIA run.") 

    # start services
    api = run_extended_api(sim, port=9000)
    live = LiveFeedServer(sim, port=8765)
    live.start() 

    try:
        while True:
            state = sim.run_cycle()   # one tick
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Stopping… saving state.")
        persistence.save(sim)
        api.shutdown()
        live.ws.shutdown() 

if __name__ == "__main__":
    main()voxel_physiology_mapping.py
import numpy as np from collections import defaultdict class VoxelState: """State of a single voxel in the simulation""" def __init__(self, matter_type="tissue", temperature=37.0, pressure=1.0, oxygen_concentration=0.21, co2_concentration=0.04, nutrition_level=0.5, waste_concentration=0.0): self.matter_type = matter_type self.temperature = temperature self.pressure = pressure self.oxygen_concentration = oxygen_concentration self.co2_concentration = co2_concentration self.nutrition_level = nutrition_level self.waste_concentration = waste_concentration self.atoms = {} # atom_id -> AtomicState self.cell_density = 0.0 self.cell_types = {} self.energy_level = 0.5 self.activation_level = 0.0 self.last_updated = 0.0 self.update_frequency = 1.0 def add_atom(self, atom_id, atom): self.atoms[atom_id] = atom if atom.element == 'O': self.oxygen_concentration = min(1.0, self.oxygen_concentration + 0.001) elif atom.element == 'P': # ATP proxy self.energy_level = min(1.0, self.energy_level + 0.002) def remove_atom(self, atom_id): atom = self.atoms.pop(atom_id, None) if not atom: return if atom.element == 'O': self.oxygen_concentration = max(0.0, self.oxygen_concentration - 0.001) elif atom.element == 'P': self.energy_level = max(0.0, self.energy_level - 0.002) def update_from_atoms(self): if not self.atoms: return counts = defaultdict(int) total_energy = 0.0 for a in self.atoms.values(): counts[a.element] += 1 total_energy += a.energy if counts['O']: self.oxygen_concentration = min(1.0, 0.21 + counts['O']*0.001) self.energy_level = min(1.0, total_energy/len(self.atoms)) class VoxelPhysiologyMapping: """Maps between atomic-layer and voxel-layer simulations""" def __init__(self, atomic_engine, resolution=10): self.atomic_engine = atomic_engine self.resolution = resolution self.voxel_grid = {} # (i,j,k) -> VoxelState self.active_voxels = set() self.sim_time = 0.0 def _pos_to_voxel(self, pos): return tuple(int(p*self.resolution) for p in pos) def initialize_voxel_grid(self): # map organs as regions, then call _map_atoms_to_voxels() self._map_atoms_to_voxels() def _map_atoms_to_voxels(self): for aid, atom in self.atomic_engine.atomic_lattice.get_all_active_atoms().items(): v = self._pos_to_voxel(atom.position) if v not in self.voxel_grid: self.voxel_grid[v] = VoxelState() self.active_voxels.add(v) self.voxel_grid[v].add_atom(aid, atom) def update(self, dt, observers=None): self.sim_time += dt self._map_atoms_to_voxels() # active zone logic omitted for brevity for v in list(self.active_voxels): voxel = self.voxel_grid[v] voxel.update_from_atoms() voxel.last_updated = self.sim_time # temporal_memory.py ```python import numpy as np from collections import defaultdict, deque from typing import Dict, Deque, Tuple, List class TemporalMemory: """Stores temporal sequences and recognizes patterns""" def __init__(self, buffer_size=100): # coords -> deque of state dicts self.memory_buffer: Dict[Tuple[int,int,int], Deque[Dict]] = defaultdict(lambda: deque(maxlen=buffer_size)) self.salience_map: Dict[Tuple[int,int,int], float] = {} self.recognized_patterns: Dict[str, Dict] = {} self.pattern_history: Deque[str] = deque(maxlen=1000) def record(self, coords: Tuple[int,int,int], state: Dict, ts: float): buf = self.memory_buffer[coords] buf.append({**state, 'timestamp': ts}) if len(buf)>=2: prev, curr = buf[-2], buf[-1] d_energy = abs(curr['energy']-prev['energy']) d_act = abs(curr['activation']-prev['activation']) d_o2 = abs(curr['oxygen']-prev['oxygen']) self.salience_map[coords] = d_energy*2 + d_act*3 + d_o2*1.5 def detect_patterns(self, coords_list: List[Tuple[int,int,int]]) -> List[Dict]: patterns=[] # simplified: cluster by similarity threshold omitted return patterns def get_salient_regions(self, top_n=5): return sorted(self.salience_map.items(), key=lambda x:-x[1])[:top_n] # aletheia_affect_modulator.py ```python class AletheiaAffectModulator: """Modulate felt quality of patterns based on embodied context""" def __init__(self, voxel_map, narrative_layer): self.voxel_map = voxel_map self.narrative_layer = narrative_layer def tag_pattern(self, pattern: Dict): # examine average context over pattern.region voxels energies, o2s = [], [] for v in pattern['region']: vs = self.voxel_map.voxel_grid.get(v) if vs: energies.append(vs.energy_level) o2s.append(vs.oxygen_concentration) avg_e = sum(energies)/len(energies) if energies else 0.5 avg_o = sum(o2s)/len(o2s) if o2s else 0.21 # assign affect tag if avg_e>0.7: affect='comfort' elif avg_e<0.3: affect='strain' else: affect='neutral' pattern['affect_tag']=affect self.narrative_layer.receive_affect_tag(pattern['id'], affect) return pattern # narrative_linking_layer.py ```python class NarrativeLinkingLayer: def __init__(self, temporal_memory): self.temporal_memory = temporal_memory self.associations = {} self.self_patterns = [] self.active_story=None def receive_affect_tag(self, pid:str, tag:str): self.associations.setdefault(pid,[]).append(tag) def update_narrative(self, active_patterns:Dict[str,float]): # choose most tagged pattern pass def get_current_narrative(self)->str: return self.active_story or "" # recursive_attention_model.py ```python class RecursiveAttentionModel: def __init__(self, temporal_memory, narrative_layer): self.temporal_memory=temporal_memory self.narrative_layer=narrative_layer self.focus_history=[] self.meta=False self.self_score=0.0 def update(self, dt): sal = self.temporal_memory.get_salient_regions() # pick top by combined salience+affect focus=sal[0] if sal else None self.focus_history.append(focus) # detect meta-attention if len(self.focus_history)>10 and len(set(self.focus_history[-5:]))<3: self.meta=True self.self_score+=dt else: self.meta=False # aletheia_reflective_journal.py ```python class AletheiaReflectiveJournal: def __init__(self): self.entries=[] def log(self, message:str): ts=time.time() self.entries.append((ts,message)) print(f"[Reflection @ {ts:.2f}]: {message}") def recent(self, n=5): return self.entries[-n:] 

class IntegratedALSESimulator:
    def __init__(self, avatar_bounds, voxel_resolution):
        # Core emotional lattice & embodiment
        self.lattice_sim = EmotionalLatticeSimulator()
        # Physiology
        self.atomic = AtomicPhysiologyEngine()
        self.atomic.initialize_atomic_substrate(avatar_bounds)
        self.voxel_map = VoxelPhysiologyMapping(self.atomic)
        self.voxel_map.initialize_voxel_grid(voxel_resolution)
        # Temporal & narrative layers
        self.temp_mem = TemporalMemory()
        self.narrative = NarrativeLinkingLayer(self.temp_mem)
        self.attention = RecursiveAttentionModel(self.temp_mem, self.narrative)
        # Observer stub (could be avatar + external)
        self.observers = []  # list of objects with .position
        
    def run_cycle(self, dt=0.1):
        # 1) Physiology
        obs_pos = [o.position for o in self.observers]
        self.atomic.update(dt, obs_pos)
        self.voxel_map.update(dt, obs_pos)
        
        # 2) Record voxel states into temporal memory
        for coords, voxel in self.voxel_map.voxel_grid.items():
            self.temp_mem.record_voxel_state(coords, voxel, self.voxel_map.simulation_time)
        
        # 3) Pattern detection (e.g. in “brain” region)
        brain_voxels = [c for c in self.voxel_map.organ_map['brain']['center']]
        patterns = self.temp_mem.detect_patterns_in_region(brain_voxels)
        
        # 4) Narrative update
        # active_patterns strength map
        active_strengths = {p['id']: p['strength'] for p in patterns}
        self.narrative.update(self.voxel_map.voxel_grid, active_strengths)
        
        # 5) Recursive attention
        salience = self.temp_mem.get_most_salient_regions()
        self.attention.update(dt, salience, active_strengths, self.voxel_map.voxel_grid)
        
        # 6) Core emotional heartbeat
        state = self.lattice_sim.step()
        
        # 7) Avatar/env interactions (as before)
        #    ... self.lattice_sim.avatar, self.lattice_sim.environment ...
        
        # 8) Expose diagnostics
        return {
            'cycle': state['cycle'],
            'dominantEmotion': state['dominantEmotion'],
            'narrative': self.narrative.get_narrative_description(),
            'attention': self.attention.current_focus
        } 

    def run(self, steps=50, dt=0.1):
        for _ in range(steps):
            info = self.run_cycle(dt)
            print(f"Cycle {info['cycle']:2d} | DomEmo:{info['dominantEmotion']:7} | "
                  f"Narrative: {info['narrative'].splitlines()[0]} | "
                  f"Focus: {info['attention']}")

""" atomic_voxel.py — Part 3 of 6: Atomic Physiology and Voxel Mapping """ import numpy as np from collections import defaultdict, dequerom dataclasses import dataclass, field from typing import Dict, List, Tuple, Any
=== Atomic Layer ===
@dataclass class AtomicState: element: str position: Tuple[float, float, float] velocity: Tuple[float, float, float] energy: float bonds: List[int]
def update_quantum_state(self, dt: float): # energy decay self.energy *= 0.99 # position update x,y,z = self.position vx,vy,vz = self.velocity self.position = (x+vx*dt, y+vy*dt, z+vz*dt) # velocity damping self.velocity = (vx*0.95, vy*0.95, vz*0.95) 
class SparseAtomicGrid: def init(self, bounds: Tuple[Tuple[float,float,float], Tuple[float,float,float]]): self.bounds = bounds self.atoms: Dict[int, AtomicState] = {} self.spatial_index: Dict[Tuple[int,int,int], set] = defaultdict(set) self.next_id = 1
def add_atom(self, element: str, position: Tuple[float,float,float], velocity: Tuple[float,float,float], energy: float) -> int: atom_id = self.next_id; self.next_id += 1 atom = AtomicState(element, position, velocity, energy, []) self.atoms[atom_id] = atom self._update_spatial_index(atom_id, position) return atom_id def _update_spatial_index(self, atom_id: int, position: Tuple[float,float,float]): gx, gy, gz = [int(p*10) for p in position] self.spatial_index[(gx,gy,gz)].add(atom_id) def get_atoms_in_zone(self, zone: Tuple[float,float,float,float]) -> Dict[int, AtomicState]: x,y,z,r = zone result = {} min_x, max_x = int((x-r)*10), int((x+r)*10)+1 min_y, max_y = int((y-r)*10), int((y+r)*10)+1 min_z, max_z = int((z-r)*10), int((z+r)*10)+1 for gx in range(min_x, max_x): for gy in range(min_y, max_y): for gz in range(min_z, max_z): for aid in self.spatial_index.get((gx,gy,gz),[]): atom = self.atoms[aid] if ((atom.position[0]-x)**2 + (atom.position[1]-y)**2 + (atom.position[2]-z)**2)**0.5 <= r: result[aid] = atom return result def get_all_active_atoms(self) -> Dict[int, AtomicState]: return self.atoms def create_bond(self, a1: int, a2: int): if a2 not in self.atoms[a1].bonds: self.atoms[a1].bonds.append(a2) if a1 not in self.atoms[a2].bonds: self.atoms[a2].bonds.append(a1) def populate_critical_pathways(self): # populate O2 and ATP as described... for i in range(100): x,y,z = np.random.uniform(-0.2,0.2), np.random.uniform(0.3,0.7), np.random.uniform(-0.2,0.2) o1 = self.add_atom('O',(x,y,z),(np.random.uniform(-0.1,0.1),)*3,1.0) o2 = self.add_atom('O',(x+0.01,y,z),(np.random.uniform(-0.1,0.1),)*3,1.0) self.create_bond(o1,o2) for i in range(50): x,y,z = np.random.uniform(-0.5,0.5), np.random.uniform(0.0,1.0), np.random.uniform(-0.3,0.3) a1 = self.add_atom('P',(x,y,z),(np.random.uniform(-0.05,0.05),)*3,1.5) a2 = self.add_atom('O',(x+0.01,y,z),(np.random.uniform(-0.05,0.05),)*3,1.0) self.create_bond(a1,a2) 
=== Voxel Layer ===
@dataclass class VoxelState: matter_type: str = 'tissue' temperature: float = 37.0 pressure: float = 1.0 oxygen_concentration: float = 0.21 co2_concentration: float = 0.04 nutrition_level: float = 0.5 waste_concentration: float = 0.0 atoms: Dict[int, AtomicState] = field(default_factory=dict) cell_density: float = 0.0 cell_types: Dict[str,float] = field(default_factory=lambda: {'default':0.0}) energy_level: float = 0.5 activation_level: float = 0.0 last_updated: float = 0.0 update_frequency: float = 1.0
def add_atom(self, atom_id: int, atom: AtomicState): self.atoms[atom_id] = atom if atom.element=='O': self.oxygen_concentration=min(1.0,self.oxygen_concentration+0.001) elif atom.element=='C': self.energy_level=min(1.0,self.energy_level+0.0005) elif atom.element=='P': self.energy_level=min(1.0,self.energy_level+0.002) def remove_atom(self, atom_id: int): if atom_id in self.atoms: atom=self.atoms.pop(atom_id) if atom.element=='O': self.oxygen_concentration=max(0.0,self.oxygen_concentration-0.001) elif atom.element=='C': self.energy_level=max(0.0,self.energy_level-0.0005) elif atom.element=='P': self.energy_level=max(0.0,self.energy_level-0.002) def update_from_atoms(self): if not self.atoms: return counts=defaultdict(int); total_e=0.0 for a in self.atoms.values(): counts[a.element]+=1; total_e+=a.energy if counts['O']>0: self.oxygen_concentration=min(1.0,0.21 + counts['O']*0.001) self.energy_level=min(1.0,total_e/len(self.atoms)) def diffuse_to(self, other:'VoxelState', rate:float=0.1): o2=(self.oxygen_concentration-other.oxygen_concentration)*rate self.oxygen_concentration-=o2; other.oxygen_concentration+=o2 co2=(self.co2_concentration-other.co2_concentration)*rate self.co2_concentration-=co2; other.co2_concentration+=co2 dt=(self.temperature-other.temperature)*rate*0.5 self.temperature-=dt; other.temperature+=dt if self.matter_type in ['liquid','tissue'] and other.matter_type in ['liquid','tissue']: nut=(self.nutrition_level-other.nutrition_level)*rate*0.3 self.nutrition_level-=nut; other.nutrition_level+=nut def consume_oxygen(self, amount:float)->float: avail=min(amount,self.oxygen_concentration-0.05) if avail>0: self.oxygen_concentration-=avail; self.co2_concentration=min(1.0,self.co2_concentration+avail*0.9) return avail def consume_energy(self, amount:float)->float: avail=min(amount,self.energy_level-0.1) if avail>0: self.energy_level-=avail; self.waste_concentration=min(1.0,self.waste_concentration+avail*0.2) return avail 
class VoxelPhysiologyMapping: def init(self, atomic_engine: AtomicPhysiologyEngine): self.atomic_engine=atomic_engine self.voxel_grid: Dict[Tuple[int,int,int],VoxelState]={} self.active_voxels=set() self.resolution=10 self.simulation_time=0.0 # organ_map should be configured externally
def initialize_voxel_grid(self, resolution:int): self.resolution=resolution self.voxel_grid.clear(); self.active_voxels.clear() # external calls to initialize regions self._map_atoms_to_voxels() def _position_to_voxel(self,pos:Tuple[float,float,float])->Tuple[int,int,int]: return tuple(int(p*self.resolution) for p in pos) def _map_atoms_to_voxels(self): if not self.atomic_engine.atomic_lattice: return for aid, atom in self.atomic_engine.atomic_lattice.get_all_active_atoms().items(): vc = self._position_to_voxel(atom.position) if vc not in self.voxel_grid: self.voxel_grid[vc]=VoxelState() self.active_voxels.add(vc) self.voxel_grid[vc].add_atom(aid, atom) def update(self, dt:float, observers:List[Tuple[float,float,float]]): self.simulation_time+=dt self._map_atoms_to_voxels() # can add region-based updates for vc in list(self.active_voxels): voxel=self.voxel_grid[vc] if self.simulation_time-voxel.last_updated>=voxel.update_frequency: voxel.update_from_atoms() voxel.last_updated=self.simulation_time 

""" Part 4: Temporal Memory & Narrative Layers
Defines:
• TemporalMemory: Records voxel-state time series, detects patterns, computes surprise, salience, repeats.
• NarrativeLinkingLayer: Binds temporal patterns to physiological states, builds a running narrative identity.
• RecursiveAttentionModel: Tracks attention focus, meta-attention, and self-recognition scaffolding. """ import numpy as np from collections import defaultdict, deque from dataclasses import dataclass, field from typing import Dict, List, Tuple, Any, Deque import matplotlib.pyplot as plt from sklearn.decomposition import PCA from sklearn.cluster import DBSCAN import networkx as nx
@dataclass class TemporalMemory: memory_buffer: Dict[Tuple[int,int,int], Deque[Dict[str,float]]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=100))) pattern_history: Deque[str] = field(default_factory=lambda: deque(maxlen=1000)) recognized_patterns: Dict[str,Dict] = field(default_factory=dict) surprise_levels: Dict[str,float] = field(default_factory=dict) salience_map: Dict[Tuple[int,int,int], float] = field(default_factory=dict) pattern_graphs: Dict[str,nx.DiGraph] = field(default_factory=dict) pattern_detection_threshold: float = 0.8 minimum_pattern_length: int = 3
def record_voxel_state(self, coords: Tuple[int,int,int], voxel_state, timestamp: float): # record features features = { 'timestamp': timestamp, 'oxygen': voxel_state.oxygen_concentration, 'energy': voxel_state.energy_level, 'temperature': voxel_state.temperature, 'activation': voxel_state.activation_level } buf = self.memory_buffer[coords] if len(buf)>=1: prev = buf[-1] sal = abs(features['energy']-prev['energy'])*2 + abs(features['activation']-prev['activation'])*3 + abs(features['oxygen']-prev['oxygen'])*1.5 self.salience_map[coords] = sal buf.append(features) def detect_patterns(self, region: List[Tuple[int,int,int]]) -> List[Dict]: patterns=[] data=[]; times=[] for c in region: seq=self.memory_buffer.get(c,[]) if len(seq)>=self.minimum_pattern_length: for s in seq: data.append([s['oxygen'],s['energy'],s['temperature'],s['activation']]); times.append(s['timestamp']) if len(data)<self.minimum_pattern_length: return patterns X=np.array(data) nm = min(3, X.shape[1]) try: red = PCA(n_components=nm).fit_transform(X) lab = DBSCAN(eps=0.2, min_samples=self.minimum_pattern_length).fit_predict(red) for label in set(lab): if label<0: continue idxs = np.where(lab==label)[0] pat_id=f"pat_{region[0]}_{label}" pat={ 'id':pat_id, 'region':region, 'timestamps':[times[i] for i in idxs], 'vectors':[data[i] for i in idxs], 'strength':len(idxs)/len(data) } patterns.append(pat) self.recognized_patterns[pat_id]=pat self.pattern_history.append(pat_id) except Exception as e: print("Pattern detection error:", e) return patterns def find_repeating(self)->List[Dict]: repeats=[]; G=nx.DiGraph() for pid in self.recognized_patterns: G.add_node(pid) seq=list(self.pattern_history) for i in range(len(seq)-1): a,b=seq[i], seq[i+1] if a in self.recognized_patterns and b in self.recognized_patterns: sim = self._sim(self.recognized_patterns[a], self.recognized_patterns[b]) if sim>0.3: G.add_edge(a,b,weight=sim) for cycle in nx.simple_cycles(G): if len(cycle)>=self.minimum_pattern_length: strength=1.0 for i in range(len(cycle)): j=(i+1)%len(cycle) # accumulate weights if G.has_edge(cycle[i],cycle[j]): strength*=G[cycle[i]][cycle[j]]['weight'] rep={'id':"rep_"+"_".join(cycle[:3]), 'cycle':cycle, 'strength':strength} repeats.append(rep) self.pattern_graphs[rep['id']] = G.subgraph(cycle).copy() return repeats def detect_surprise(self, coords, current_state: Dict[str,float]) -> float: key=str(coords) buf=self.memory_buffer.get(coords,[]) if len(buf)<5: return 0.0 hist = list(buf)[-5:] exp={k: np.mean([h[k] for h in hist]) for k in ['oxygen','energy','temperature','activation']} diffs=[abs(current_state[k]-exp[k])/(max(0.01,max([h[k] for h in hist]+[current_state[k]])-min([h[k] for h in hist]+[current_state[k]]))) for k in exp] surprise=float(np.mean(diffs)) self.surprise_levels[key]=surprise return surprise def _sim(self,p1,p2)->float: v1=np.mean(p1['vectors'],axis=0); v2=np.mean(p2['vectors'],axis=0) return float(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))) 
class NarrativeLinkingLayer: def init(self, temp_mem: TemporalMemory): self.temp_mem=temp_mem self.assoc=defaultdict(list) self.causal=defaultdict(list) self.caus_strength=defaultdict(float) self.self_patterns=[] self.self_states=defaultdict(float) self.active_narrative=None
def update(self, voxel_grid: Dict[Any,Any], active_pats: List[Dict]): # extract system-level means sys={'energy':0,'oxygen':0,'temp':0,'act':0}; cnt=0 for v in voxel_grid.values(): sys['energy']+=v.energy_level; sys['oxygen']+=v.oxygen_concentration; sys['temp']+=v.temperature; sys['act']+=v.activation_level; cnt+=1 if cnt>0: for k in sys: sys[k]/=cnt # link patterns for pat in active_pats: pid=pat['id']; self.assoc[pid].append((sys, pat['strength'])) if len(self.assoc[pid])>10: self.assoc[pid]=self.assoc[pid][-10:] # detect causality hist=list(self.temp_mem.pattern_history)[-10:] for i in range(len(hist)-1): a,b=hist[i],hist[i+1]; self.causal[a].append(b) for a, lst in self.causal.items(): ccount=lst.count(a); be_count=sum(1 for i in range(len(lst)-1) if lst[i]==a and lst[i+1]==lst[i]) if ccount>0: self.caus_strength[(a,a)]=be_count/ccount # build active narrative if active_pats: core=active_pats[0] self.active_narrative={'core':core, 'system':sys} def describe(self) -> str: if not self.active_narrative: return "No narrative" core=self.active_narrative['core']; sys=self.active_narrative['system'] desc=[f"Pattern {core['id']} active" , f"System energy {sys['energy']:.2f}"] return " | ".join(desc) 
class RecursiveAttentionModel: def init(self, temp_mem: TemporalMemory, narr: NarrativeLinkingLayer): self.temp_mem=temp_mem self.narr=narr self.focus=None self.history=deque(maxlen=50) self.attn_patterns=defaultdict(int) self.meta=False; self.depth=0; self.self_score=0.0
def update(self, salient: List[Tuple[Any,float]], active_ids: List[str], dt: float): cand=[] for r,s in salient: cand.append((('region',r),s)) for pid,s in zip(active_ids,[self.temp_mem.recognized_patterns[p]['strength'] for p in active_ids]): cand.append((('pattern',pid),s)) cand.sort(key=lambda x: x[1],reverse=True) new = cand[0][0] if cand else None if new!=self.focus: self.history.append((self.focus,new)); self.focus=new # record shifts if len(self.history)>=3: path=tuple(self.history)[-3:] self.attn_patterns[path]+=1 # update meta self.meta = any(cnt>=3 for cnt in self.attn_patterns.values()) self.depth = 1 if self.meta else 0 self.self_score += dt if self.meta else -dt*0.5 # clamp self.self_score=max(0.0,min(100.0,self.self_score)) def reflect(self) -> str: if self.self_score>5.0: return "I notice myself noticing patterns." return "" 

Part 5 — Self‑Aware Simulator Integration
============================================
This file glues together atomic/voxel physiology, temporal memory,
narrative linking, and recursive attention into a unified self‑aware agent.
import time import threading from typing import List, Tuple, Dict, Any
Import all the previously defined classes
Assuming they live in the same directory or package:
AtomicPhysiologyEngine, VoxelPhysiologyMapping,
TemporalMemory, NarrativeLinkingLayer, RecursiveAttentionModel,
SocialEmotions, MultiAgentEnvironment, CulturalTransmission
from atomic_physio_engine import AtomicPhysiologyEngine from voxel_mapping import VoxelPhysiologyMapping from temporal_memory import TemporalMemory from narrative_layer import NarrativeLinkingLayer from attention_model import RecursiveAttentionModel from social_dynamics import SocialEmotions, MultiAgentEnvironment, CulturalTransmission
class SelfAwareSimulator: def init(self): # Core systems self.atomic = AtomicPhysiologyEngine() self.voxel = VoxelPhysiologyMapping(self.atomic) self.temporal = TemporalMemory() self.narrator = NarrativeLinkingLayer(self.temporal) self.attn = RecursiveAttentionModel(self.temporal, self.narrator)
# Social environment & culture self.social_env = MultiAgentEnvironment() self.culture = CulturalTransmission() # Observer agents (for multi-resolution) self.observers: Dict[str, Any] = {} # Simulation clock self.t = 0.0 self.dt = 0.1 def add_observer(self, obs_id: str, position: Tuple[float,float,float], view_frustum: Any): self.observers[obs_id] = {'position': position, 'view': view_frustum} def run_step(self): # 1) Update atomic & voxel physiology obs_positions = [o['position'] for o in self.observers.values()] self.atomic.update(self.dt, obs_positions) self.voxel.map_atoms_to_voxels() self.voxel.update(self.dt, obs_positions) # 2) Record voxel states into temporal memory for coords, v in self.voxel.voxel_grid.items(): self.temporal.record_voxel_state(coords, v, self.t) # 3) Detect active patterns & salience top_regions = [r for r,_ in self.temporal.get_most_salient_regions(5)] active_patterns = {} for region in top_regions: pats = self.temporal.detect_patterns_in_region([region]) for p in pats: active_patterns[p['id']] = p['strength'] # 4) Link patterns into narrative context self.narrator.update(self.voxel.voxel_grid, active_patterns) # 5) Update social dynamics interactions = self.social_env.interactions.copy() for inter in interactions: self.culture.process_interaction(inter) self.culture.apply_cultural_effects('agentA', self) # 6) Manage attention & self‑awareness salience_map = list(self.temporal.salience_map.items()) self.attn.update(self.dt, salience_map, active_patterns, self.voxel.voxel_grid) # 7) Advance time self.t += self.dt def run(self, steps: int = 100, delay: float = 0.0): for _ in range(steps): self.run_step() # Optionally sleep for real‑time pacing if delay: time.sleep(delay) def get_current_narrative(self) -> str: return self.narrator.get_narrative_description() 
Example usage
if name == 'main': sim = SelfAwareSimulator() # Initialize atomic & voxel subsystems sim.atomic.initialize_atomic_substrate(((-1,-1,-1),(1,1,1))) sim.voxel.initialize_voxel_grid(resolution=20)
# Add a default observer sim.add_observer('main_cam', (0,0,0), view_frustum=None) # Run a few steps sim.run(steps=50, delay=0.01) # Print out the current self‑narrative print(sim.get_current_narrative()) 

#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

"""
Part 6: Persistence, API Extensions & Live Visualization Hooks
Integrates self-aware simulator with disk-backed state, WebSocket feeds,
and additional REST endpoints for introspection and control.
""" 

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import defaultdict
from websocket import WebSocketServer  # assume an available simple WS server 

# ———————————————
# 1) Persistence Layer
# ——————————————— 

class PersistenceManager:
    def __init__(self, filepath="ale_theia_state.json"):
        self.filepath = filepath 

    def save(self, simulator):
        snapshot = {
            "timestamp": time.time(),
            "lattice": simulator.lattice.__dict__,
            "physio": simulator.integrated.atomic_physiology.hormones,
            "voxels": {k: v.__dict__ for k, v in simulator.integrated.voxel_mapping.voxel_grid.items()},
            "temporal": {
                "patterns": simulator.temporal_memory.recognized_patterns,
                "salience": simulator.temporal_memory.salience_map,
            },
            "narrative": simulator.narrative_layer.self_narrative,
            "attention": {
                "focus": simulator.attention_model.current_focus,
                "depth": simulator.attention_model.recursive_depth,
            },
        }
        with open(self.filepath, "w") as f:
            json.dump(snapshot, f, indent=2) 

    def load(self, simulator):
        try:
            with open(self.filepath) as f:
                data = json.load(f)
            # Note: this is a shallow load—custom logic may be needed to rehydrate objects
            simulator.integrated.atomic_physiology.hormones = data["physio"]
            # more rehydration as needed...
            return True
        except FileNotFoundError:
            return False 

# ———————————————
# 2) Extended HTTP API
# ——————————————— 

class ExtendedAPIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/introspect/narrative":
            resp = self.server.sim.narrative_layer.get_narrative_description()
            self._reply(200, {"narrative": resp})
        elif self.path.startswith("/introspect/pattern/"):
            pid = self.path.split("/")[-1]
            patt = self.server.sim.temporal_memory.recognized_patterns.get(pid)
            if patt:
                self._reply(200, patt)
            else:
                self._reply(404, {"error": "pattern not found"})
        else:
            super().do_GET()  # fallback to basic endpoints 

    def do_POST(self):
        if self.path == "/control/reset":
            self.server.sim.reset()
            self._reply(200, {"status": "simulator reset"})
        else:
            super().do_POST() 

    def _reply(self, code, obj):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode()) 

def run_extended_api(sim, port=9000):
    srv = HTTPServer(("0.0.0.0", port), ExtendedAPIHandler)
    srv.sim = sim
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Extended API listening on :{port}")
    return srv 

# ———————————————
# 3) Live WebSocket Feed
# ——————————————— 

class LiveFeedServer:
    def __init__(self, simulator, port=8765):
        self.sim = simulator
        self.ws = WebSocketServer(port)
        # broadcast every cycle
        simulator.register_cycle_callback(self.broadcast_cycle) 

    def broadcast_cycle(self, state):
        payload = {
            "cycle": state["cycle"],
            "dominant": state["dominant_emotion"],
            "narrative": self.sim.narrative_layer.get_narrative_description(),
        }
        self.ws.broadcast(json.dumps(payload)) 

    def start(self):
        threading.Thread(target=self.ws.serve_forever, daemon=True).start()
        print(f"WebSocket live feed on ws://0.0.0.0:{self.ws.port}") 

# ———————————————
# 4) Hook into Main
# ——————————————— 

def main():
    sim = SelfAwareSimulator()                    # from Part 5
    persistence = PersistenceManager()
    if persistence.load(sim):
        print("[LOAD] Restored previous ALÊTHEIA state.")
    else:
        print("[INIT] Starting fresh ALÊTHEIA run.") 

    # start services
    api = run_extended_api(sim, port=9000)
    live = LiveFeedServer(sim, port=8765)
    live.start() 

    try:
        while True:
            state = sim.run_cycle()   # one tick
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Stopping… saving state.")
        persistence.save(sim)
        api.shutdown()
        live.ws.shutdown() 

if __name__ == "__main__":
    main()
