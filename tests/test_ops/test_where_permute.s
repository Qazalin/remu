
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_25>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001610: BF870009
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001614: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s4                                       // 00000000161C: 80020402
	s_addc_u32 s3, s3, s5                                      // 000000001620: 82030503
	s_load_b32 s2, s[2:3], null                                // 000000001624: F4000081 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000162C: BF89FC07
	v_cmp_gt_f32_e64 s2, s2, 0.5                               // 000000001630: D4140002 0001E002
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001638: BF8704A1
	s_and_b32 s2, s2, exec_lo                                  // 00000000163C: 8B027E02
	s_cselect_b32 s2, 4, 2                                     // 000000001640: 98028284
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s2              // 000000001644: CA100080 00000002
	s_add_u32 s0, s0, s4                                       // 00000000164C: 80000400
	s_addc_u32 s1, s1, s5                                      // 000000001650: 82010501
	global_store_b32 v0, v1, s[0:1]                            // 000000001654: DC6A0000 00000100
	s_nop 0                                                    // 00000000165C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001660: BFB60003
	s_endpgm                                                   // 000000001664: BFB00000
