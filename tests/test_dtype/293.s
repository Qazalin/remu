
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_10n105>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	v_mov_b32_e32 v2, 0                                        // 000000001610: 7E040280
	s_lshl_b64 s[6:7], s[4:5], 2                               // 000000001614: 84868204
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s6                                       // 00000000161C: 80020602
	s_addc_u32 s3, s3, s7                                      // 000000001620: 82030703
	s_load_b32 s6, s[2:3], null                                // 000000001624: F4000181 F8000000
	s_lshl_b64 s[2:3], s[4:5], 3                               // 00000000162C: 84828304
	s_waitcnt lgkmcnt(0)                                       // 000000001630: BF89FC07
	s_ashr_i32 s4, s6, 31                                      // 000000001634: 86049F06
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001638: BF870009
	v_dual_mov_b32 v0, s6 :: v_dual_mov_b32 v1, s4             // 00000000163C: CA100006 00000004
	s_add_u32 s0, s0, s2                                       // 000000001644: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001648: 82010301
	global_store_b64 v2, v[0:1], s[0:1]                        // 00000000164C: DC6E0000 00000002
	s_nop 0                                                    // 000000001654: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001658: BFB60003
	s_endpgm                                                   // 00000000165C: BFB00000
