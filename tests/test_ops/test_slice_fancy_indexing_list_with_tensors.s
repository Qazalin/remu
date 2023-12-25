
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_2_2n2>:
	s_load_b64 s[0:1], s[0:1], null                            // 000000001600: F4040000 F8000000
	s_sub_i32 s3, 0, s15                                       // 000000001608: 81830F80
	s_not_b32 s4, s15                                          // 00000000160C: BE841E0F
	s_lshr_b32 s3, s3, 31                                      // 000000001610: 85039F03
	s_lshr_b32 s4, s4, 31                                      // 000000001614: 85049F04
	s_mov_b32 s2, s15                                          // 000000001618: BE82000F
	s_add_i32 s4, s3, s4                                       // 00000000161C: 81040403
	s_ashr_i32 s3, s15, 31                                     // 000000001620: 86039F0F
	s_add_i32 s4, s4, -1                                       // 000000001624: 8104C104
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001628: 84828202
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s4              // 00000000162C: CA100080 00000004
	s_waitcnt lgkmcnt(0)                                       // 000000001634: BF89FC07
	s_add_u32 s0, s0, s2                                       // 000000001638: 80000200
	s_addc_u32 s1, s1, s3                                      // 00000000163C: 82010301
	global_store_b32 v0, v1, s[0:1]                            // 000000001640: DC6A0000 00000100
	s_nop 0                                                    // 000000001648: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000164C: BFB60003
	s_endpgm                                                   // 000000001650: BFB00000
