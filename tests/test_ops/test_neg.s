
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_2925n26>:
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
	s_xor_b32 s2, s2, 0x80000000                               // 000000001630: 8D02FF02 80000000
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001638: BF870009
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s2              // 00000000163C: CA100080 00000002
	s_add_u32 s0, s0, s4                                       // 000000001644: 80000400
	s_addc_u32 s1, s1, s5                                      // 000000001648: 82010501
	global_store_b32 v0, v1, s[0:1]                            // 00000000164C: DC6A0000 00000100
	s_nop 0                                                    // 000000001654: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001658: BFB60003
	s_endpgm                                                   // 00000000165C: BFB00000
