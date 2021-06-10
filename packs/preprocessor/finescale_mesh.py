from .. import inputs
import numpy as np
import line_profiler

@profile
def get_transmissibilities_and_adjacencies():
    finescale_inputs = inputs.finescale_inputs
    GID_0, volume_centroids, face_areas, face_adjacencies = generate_centroids_areas_and_adjacencies(finescale_inputs['mesh_generation_parameters'])
    volume_permeabilities, porosities = get_permeability_and_porosity(volume_centroids, finescale_inputs['rock_parameters'], (finescale_inputs['mesh_generation_parameters']['block_size']))
    face_permeabilities = get_k_harm_faces(volume_centroids, face_adjacencies, face_areas, volume_permeabilities)
    wells=get_wells(finescale_inputs['wells'], volume_centroids, GID_0)
    return wells, GID_0, face_adjacencies, face_permeabilities

def generate_centroids_areas_and_adjacencies(mesh):
    nb=mesh['n_blocks']
    lb=mesh['block_size']
    sp=mesh['starting_point']
    ms = np.mgrid[sp[0]+0.5*lb[0]:sp[0]+(nb[0]+0.5)*lb[0]:lb[0],
                  sp[1]+0.5*lb[1]:sp[1]+(nb[1]+0.5)*lb[1]:lb[1],
                  sp[2]+0.5*lb[2]:sp[2]+(nb[2]+0.5)*lb[2]:lb[2]]
    ms = ms.flatten()
    centroids = ms.reshape(3,int(ms.size/3)).T
    GID_0 = np.arange(len(centroids))
    adjs0 = np.tile(GID_0,3)
    adjs1 = np.concatenate([GID_0+1, GID_0+nb[2], GID_0+nb[1]*nb[2]])
    adjs=np.vstack([adjs0,adjs1]).T
    adjs=adjs[(adjs.min(axis=1)>=0) & (adjs.max(axis=1)<=GID_0.max())]
    dif=abs(centroids[adjs[:,0]]-centroids[adjs[:,1]])
    adjacencies=adjs[((dif>0).sum(axis=1)==1) & ((dif<=lb).sum(axis=1)==3)]
    areas=np.array([lb[1]*lb[2], lb[0]*lb[2], lb[0]*lb[1]])
    areas=np.tile(areas,len(adjacencies)).reshape(len(adjacencies),3)[abs(centroids[adjacencies[:,0]]-centroids[adjacencies[:,1]])>0]
    return GID_0, centroids, areas, adjacencies

def get_permeability_and_porosity(centroids, rock, block_dim):
    if not (rock['constant_porosity'] or rock['constant_permeability']):
        file_infos=rock['file_infos']
        data_spe10 = np.load(file_infos['name'])
        ks = data_spe10['perms'][:,[0,4,8]]
        phi = data_spe10['phi']
        phi = phi.flatten()
        nx , ny, nz = file_infos['nblocks']
        ijk0 = np.array([centroids[:, 0]//block_dim[0], centroids[:, 1]//block_dim[1], centroids[:, 2]//block_dim[2]])
        ee = ijk0[0]*ny*nz + ijk0[1]*nz + ijk0[2]-1
        ee = ee.astype(np.int32)
        permeabilities=ks[ee]
        porosities=phi[ee]

    if rock['constant_porosity']:
        porosities=rock['porosity_value']
    if rock['constant_permeability']:
        permeabilities=rock['permeability_value']
    return permeabilities, porosities

def get_k_harm_faces(centroids, adjs, areas, ks):
    dist=centroids[adjs[:,1]]-centroids[adjs[:,0]]
    aux=dist>0
    l=dist.max(axis=1)
    k0 = ks[adjs[:,0]][aux]
    k1 = ks[adjs[:,1]][aux]
    k_harm = 2*k0*k1/(k0*l+k1*l)
    return k_harm

def get_wells(all_wells, centroids, GID_0):
    wells={}
    injectors=[]
    producers=[]
    diric=[]
    val_diric=[]
    neum=[]
    val_neum=[]
    for w in all_wells:
        well=all_wells[w]
        p0=well['p0']
        p1=well['p1']
        vols = GID_0[((centroids>p0) & (centroids<p1)).sum(axis=1)==3]
        if len(vols>0):
            if well['type'] == 'Injector':
                injectors.append(vols)
            elif well['type'] == 'Producer':
                producers.append(vols)

            if well['prescription'] == 'P':
                diric.append(vols)
                val_diric.append(np.repeat(well['value'],len(vols)))
            elif well['prescription'] == 'Q':
                neum.append(vols)
                val_neum.append(well['value'])

    wells['ws_inj'] = np.concatenate(injectors)
    wells['ws_prod'] = np.concatenate(producers)
    wells['ws_p'] = np.concatenate(diric)
    wells['values_p'] = np.concatenate(val_diric)
    if len(neum)>0:
        wells['ws_q'] = np.concatenate(neum)
        wells['values_q'] = np.concatenate(val_neum)
    else:
        wells['ws_q'] = np.array([])
        wells['values_q'] = np.array([])    
    return wells
