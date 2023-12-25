
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_2925n32>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001610: BF870009
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001614: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s4                                       // 00000000161C: 80020402
	s_addc_u32 s3, s3, s5                                      // 000000001620: 82030503
	s_add_u32 s0, s0, s4                                       // 000000001624: 80000400
	s_load_b32 s2, s[2:3], null                                // 000000001628: F4000081 F8000000
	s_addc_u32 s1, s1, s5                                      // 000000001630: 82010501
	s_waitcnt lgkmcnt(0)                                       // 000000001634: BF89FC07
	v_max_f32_e64 v0, s2, s2                                   // 000000001638: D5100000 00000402
	v_add_f32_e64 v1, 0xc0c00000, s2                           // 000000001640: D5030001 000004FF C0C00000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000164C: BF870091
	v_dual_max_f32 v0, 0, v0 :: v_dual_max_f32 v1, 0, v1       // 000000001650: CA940080 00000280
	v_dual_sub_f32 v0, v0, v1 :: v_dual_mov_b32 v1, 0          // 000000001658: C9500300 00000080
	global_store_b32 v1, v0, s[0:1]                            // 000000001660: DC6A0000 00000001
	s_nop 0                                                    // 000000001668: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000166C: BFB60003
	s_endpgm                                                   // 000000001670: BFB00000
