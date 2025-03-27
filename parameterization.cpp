#include <pmp/algorithms/parameterization.h>
#include "pmp/io/io.h"
#include "pmp/io/io_flags.h"
#include "pmp/io/write_obj.h"
#include <pmp/surface_mesh.h>
#include <iostream>
#include "pmp/io/read_obj.h"

  
  using namespace pmp;
  
  class ParametrizationProcessor
  {
  public:
    ParametrizationProcessor() {}
    
    void process(const char* input_filename);
    
  private:
    void performLSCMParametrization(SurfaceMesh& mesh);
    std::string filename; // Add filename as a member variable
  };


// première partie : juste pour faire un essai sur la fonction apparemment et éviter erreurs 
void ParametrizationProcessor::performLSCMParametrization(SurfaceMesh& mesh)
  {
    try
    {
      lscm_parameterization(mesh);
    }
    catch (const std::exception& e)
    {
      std::cerr << "Error during LSCM parametrization: " << e.what() << std::endl;
      return;
    }}

// 2ème partie : on produit le résultat 
void ParametrizationProcessor::process(const char* input_filename)
    {

        filename = input_filename; // Set the member variable with the provided filename
      SurfaceMesh mesh;
      
      std::cerr << "Paramétrisation en cours pour le fichier: " << input_filename << std::endl;
      
      pmp::read_obj(mesh, filename);
      
      // alloc tex coordinates
      mesh.vertex_property<TexCoord>("v:tex", TexCoord(0, 0));
      
      // Perform parametrizations
      performLSCMParametrization(mesh);
      
      // Utilisation du nom de fichier d'entrée comme base pour le fichier de sortie
      std::filesystem::path output_directory("output_lscm");
      std::filesystem::create_directory(output_directory);
      std::string base_filename = std::filesystem::path(filename).stem().string();
      std::string output_filename = (output_directory / (base_filename + "_lscm.obj")).string();
      
      // Ajout d'un numéro séquentiel pour éviter les conflits
      size_t file_number = 1;
      while (std::filesystem::exists(output_filename))
      {
        output_filename = (output_directory / (std::filesystem::path(filename).stem().string() +
          "_" + std::to_string(file_number) +
          std::filesystem::path(filename).extension().string())).string();
        ++file_number;
      }
      
      IOFlags flags;
      flags.use_vertex_texcoords = true;
      flags.use_halfedge_texcoords = true;
      flags.use_vertex_normals = true;
      flags.use_face_normals = true;
      flags.use_vertex_colors = true;
      flags.use_face_colors = true;
      
      write_obj(mesh, output_filename, flags);
      
      std::cerr << "Paramétrisation réussie pour le fichier: " << output_filename << std::endl;
      
     } 
  
  int main(int argc, char** argv)
  {
    ParametrizationProcessor processor;
    
    if (argc == 2)
      processor.process(argv[1]);
    else
      std::cerr << "Usage: " << argv[0] << " <input_mesh>" << std::endl;
    
    return 0;
  }

 
